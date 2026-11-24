"""Perceptual reconstruction losses for StoSAVi BEV pretraining.

Variants:
- lpips_vgg:
    Standard LPIPS with the VGG16 backbone (Zhang et al. 2018).

- feat_l2_vgg16_relu2_2:
    Johnson-style VGG16 feature reconstruction loss using relu2_2.
    Intended to preserve relatively low-level visual structure.

- feat_l2_vgg16_relu1_2_relu2_2:
    VGG16 feature reconstruction loss using both relu1_2 and relu2_2.
    More strongly biased toward preserving small, spatially precise details.

- feat_l2_robust_r50:
    Raw feature L2 through an adversarially-robust ImageNet ResNet-50
    (Salman et al. 2020, L2 eps=3.0). No calibrated per-layer weights;
    uniform average across layer1..layer4.

All backbones are frozen (requires_grad=False, .eval() forced) and accept
inputs in [0, 1] with shape [N, 3, H, W]. Caller is responsible for flattening
temporal batches ([B, T, 3, H, W] -> [B*T, 3, H, W]) before calling.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

_logger = logging.getLogger(__name__)


class _FrozenBackbone(nn.Module):
    """Base: freezes params, forces .eval() regardless of surrounding train()."""

    def train(self, mode: bool = True):
        return super().train(False)


class _VGG16Features(_FrozenBackbone):
    """ImageNet-pretrained VGG16 intermediate features.

    torchvision.models.vgg16().features indices:
        3: relu1_2
        8: relu2_2
    """

    def __init__(self, layers):
        super().__init__()

        valid_layers = {"relu1_2", "relu2_2"}
        unknown = set(layers) - valid_layers
        if unknown:
            raise ValueError(f"Unknown VGG16 feature layers: {unknown}")

        self.layers = tuple(layers)

        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        ).features

        # We never need anything after relu2_2.
        self.features = vgg[:9]

        self._layer_indices = {
            3: "relu1_2",
            8: "relu2_2",
        }

        for p in self.parameters():
            p.requires_grad_(False)

        self.eval()

    def forward(self, x):
        outputs = {}

        for i, layer in enumerate(self.features):
            x = layer(x)

            if i in self._layer_indices:
                name = self._layer_indices[i]
                if name in self.layers:
                    outputs[name] = x

        return [outputs[name] for name in self.layers]


class _RobustR50Features(_FrozenBackbone):
    """ResNet-50 with adversarially-robust ImageNet weights (Salman et al. 2020,
    L2 eps=3.0). Extracts features from layer1..layer4."""

    def __init__(self, ckpt_path: str):
        super().__init__()

        resnet = torchvision.models.resnet50(weights=None)

        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        cleaned = {}
        for k, v in state.items():
            for prefix in (
                "module.model.",
                "module.attacker.model.",
                "attacker.model.",
                "module.",
                "model.",
            ):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break

            if k.startswith("normalizer.") or k.startswith("attacker."):
                continue

            cleaned[k] = v

        missing, unexpected = resnet.load_state_dict(cleaned, strict=False)

        if missing or unexpected:
            _logger.info(
                "Robust R50 load: missing=%d unexpected=%d (first missing: %s)",
                len(missing),
                len(unexpected),
                missing[:3],
            )

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return [f1, f2, f3, f4]


class PerceptualLoss(nn.Module):
    """Perceptual reconstruction loss.

    Args:
        kind:
            "lpips_vgg"
            "feat_l2_vgg16_relu2_2"
            "feat_l2_vgg16_relu1_2_relu2_2"
            "feat_l2_robust_r50"

        robust_ckpt:
            Path to the Salman et al. L2 eps=3.0 ResNet-50 checkpoint.
            Required iff kind="feat_l2_robust_r50".
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, kind: str, robust_ckpt: str = None):
        super().__init__()

        self.kind = kind

        if kind == "lpips_vgg":
            try:
                import lpips
            except ImportError as e:
                raise ImportError(
                    "PerceptualLoss(kind='lpips_vgg') requires the `lpips` package. "
                    "Install with `pip install lpips`."
                ) from e

            self.backbone = lpips.LPIPS(
                net="vgg",
                verbose=False,
            )

            for p in self.backbone.parameters():
                p.requires_grad_(False)

            self.backbone.eval()

        elif kind == "feat_l2_vgg16_relu2_2":
            self.backbone = _VGG16Features(
                layers=("relu2_2",)
            )

        elif kind == "feat_l2_vgg16_relu1_2_relu2_2":
            self.backbone = _VGG16Features(
                layers=("relu1_2", "relu2_2")
            )

        elif kind == "feat_l2_robust_r50":
            if robust_ckpt is None:
                raise ValueError(
                    "PerceptualLoss(kind='feat_l2_robust_r50') requires "
                    "robust_ckpt pointing at the Salman et al. 2020 "
                    "L2 eps=3.0 ResNet-50 checkpoint."
                )

            self.backbone = _RobustR50Features(robust_ckpt)

        else:
            raise ValueError(
                f"Unknown PerceptualLoss kind: {kind!r}"
            )

        self.register_buffer(
            "_mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _normalize_imagenet(self, x):
        return (
            x - self._mean.to(dtype=x.dtype, device=x.device)
        ) / self._std.to(dtype=x.dtype, device=x.device)

    @staticmethod
    def _feature_l2(pred_feats, target_feats):
        """Uniform mean of per-layer feature MSEs."""
        loss = pred_feats[0].new_zeros(())

        for pf, tf in zip(pred_feats, target_feats):
            loss = loss + F.mse_loss(pf, tf)

        return loss / len(pred_feats)

    def forward(self, pred, target):
        pred = pred.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)

        # LPIPS performs its own input preprocessing.
        if self.kind == "lpips_vgg":
            return self.backbone(
                pred,
                target,
                normalize=True,
            ).mean()

        # VGG16 and robust ResNet branches use ImageNet normalization.
        pred_n = self._normalize_imagenet(pred)
        target_n = self._normalize_imagenet(target)

        # No gradient is required through the target branch.
        with torch.no_grad():
            target_feats = self.backbone(target_n)

        # Backbone parameters are frozen, but gradients must propagate
        # through its operations back to `pred`.
        pred_feats = self.backbone(pred_n)

        return self._feature_l2(
            pred_feats,
            target_feats,
        )


def build_perceptual_loss(kind, robust_ckpt=None):
    """Factory. kind='none' returns None."""
    if kind in (None, "none", ""):
        return None

    return PerceptualLoss(
        kind=kind,
        robust_ckpt=robust_ckpt,
    )
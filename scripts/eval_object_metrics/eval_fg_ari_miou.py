"""Compute FG-ARI and mIoU for a SAVi checkpoint on the manifest's 72 frames.

Two GT modes:
    --gt-dir <instance masks>   -> FG-ARI + Hungarian mIoU (instance-level)
    --semantic-gt-dir <vehicle-only binary masks>
                                 -> Class-IoU for the vehicle class:
                                    the model's slots are argmaxed, then any
                                    slot with substantial overlap with the
                                    GT vehicle mask is unioned into the
                                    prediction; standard IoU is reported.

Reads:
    - Manifest from prepare_test_manifest.py.
    - GT masks from generate_gt_masks.py (instance) and/or
      generate_semantic_masks.py (semantic).
    - SAVi checkpoint (state_dict).

Uses the same vp_utils primitives that SlotFormer's video-prediction eval uses.

Example (instance):
    python eval_fg_ari_miou.py \\
        --params slotformer/bev_slots/configs/savi_bev_params.py \\
        --weight $PRETRAINED_SAVI \\
        --manifest manifest.json \\
        --gt-dir gt_masks \\
        --out results_pretrained.json

Example (with semantic addendum):
    python eval_fg_ari_miou.py ... \\
        --gt-dir gt_masks \\
        --semantic-gt-dir semantic_masks \\
        --out results.json
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "slotformer" / "video_prediction"))

from slotformer.bev_slots.models import build_model  # noqa: E402
from vp_utils import postproc_mask, fARI_metric, miou_metric  # noqa: E402


def load_params(params_path: str):
    if params_path.endswith(".py"):
        params_path = params_path[:-3]
    sys.path.insert(0, os.path.dirname(os.path.abspath(params_path)))
    return importlib.import_module(os.path.basename(params_path)).SlotFormerParams()


def load_frames(route: Path, frame_indices, transform):
    imgs = []
    for f in frame_indices:
        img = Image.open(route / "birdview" / f"{f:04d}.jpg").convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs, dim=0)


def vehicle_class_iou(gt_binary: torch.Tensor, pred_argmax: torch.Tensor) -> float:
    """Union all predicted slots that overlap the GT vehicle mask,
    then report standard IoU. Handles the many-to-one slot-to-class mapping."""
    B = gt_binary.shape[0]
    ious = []
    for b in range(B):
        gt = gt_binary[b].bool()
        pr = pred_argmax[b]
        if gt.sum() == 0:
            ious.append(1.0 if pr.numel() == 0 else 0.0)
            continue
        # Union slots that have >30% of their pixels overlapping GT vehicle.
        slot_ids = torch.unique(pr)
        pred_union = torch.zeros_like(gt)
        for sid in slot_ids.tolist():
            slot_mask = (pr == sid)
            slot_area = slot_mask.sum().item()
            if slot_area == 0:
                continue
            overlap = (slot_mask & gt).sum().item()
            if overlap / slot_area > 0.3:
                pred_union |= slot_mask
        inter = (pred_union & gt).sum().item()
        union = (pred_union | gt).sum().item()
        ious.append(inter / union if union > 0 else 0.0)
    return float(np.mean(ious))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True, type=str)
    parser.add_argument("--weight", required=True, type=str)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="Instance mask directory (from generate_gt_masks.py).")
    parser.add_argument("--semantic-gt-dir", type=Path, default=None,
                        help="Semantic (vehicle-only) mask directory (from generate_semantic_masks.py).")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.gt_dir is None and args.semantic_gt_dir is None:
        raise SystemExit("Provide at least one of --gt-dir or --semantic-gt-dir.")

    params = load_params(args.params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(params)
    ckpt = torch.load(args.weight, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:3]}")
    model.testing = False
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(params.resolution),
        transforms.ToTensor(),
    ])

    manifest = json.loads(args.manifest.read_text())
    per_clip = []

    for entry in manifest["entries"]:
        obj3d_id = entry["obj3d_id"]
        route = Path(entry["route_path"])
        frames = entry["frame_indices"]

        video = load_frames(route, frames, transform).to(device)
        out_dict = model({"img": video[None]})
        pred_masks_raw = out_dict["post_masks"]              # [1, T, N, 1, H, W]
        pred_masks = postproc_mask(pred_masks_raw).cpu().long()  # [1, T, H, W]

        rec = {
            "obj3d_id": obj3d_id,
            "route_name": entry["route_name"],
            "n_frames": len(frames),
        }

        if args.gt_dir is not None:
            gt = np.load(args.gt_dir / f"{obj3d_id}_masks.npy").astype(np.int64)
            gt_t = torch.from_numpy(gt).unsqueeze(0)  # [1, T, H, W]
            gt_bt = gt_t.reshape(-1, gt_t.shape[-2], gt_t.shape[-1]).long()
            pr_bt = pred_masks.reshape(-1, pred_masks.shape[-2], pred_masks.shape[-1]).long()
            rec["fg_ari"] = float(fARI_metric(gt_bt, pr_bt))
            rec["miou"]   = float(miou_metric(gt_bt, pr_bt))
            rec["n_gt_instances_max"] = int(gt_t.max())

        if args.semantic_gt_dir is not None:
            sem = np.load(args.semantic_gt_dir / f"{obj3d_id}_semantic.npy")
            sem_t = torch.from_numpy(sem.astype(np.int64)).unsqueeze(0)  # [1, T, H, W]
            # Assume vehicle-only binary; if it's multi-class, threshold to class 10.
            if sem_t.max() > 1:
                sem_t = (sem_t == 10).long()
            sem_bt = sem_t.reshape(-1, sem_t.shape[-2], sem_t.shape[-1])
            pr_bt = pred_masks.reshape(-1, pred_masks.shape[-2], pred_masks.shape[-1])
            rec["vehicle_iou_semantic"] = vehicle_class_iou(sem_bt, pr_bt)

        per_clip.append(rec)
        line = "  " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in rec.items()
                                if k not in {"obj3d_id", "route_name", "n_frames"})
        print(f"  {obj3d_id:>3s} {line}")

    def agg(key):
        vs = [c[key] for c in per_clip if key in c]
        return {"mean": float(np.mean(vs)), "std": float(np.std(vs))} if vs else None

    result = {
        "weight": args.weight,
        "params": args.params,
        "manifest": str(args.manifest),
        "gt_dir": str(args.gt_dir) if args.gt_dir else None,
        "semantic_gt_dir": str(args.semantic_gt_dir) if args.semantic_gt_dir else None,
        "n_clips": len(per_clip),
        "fg_ari": agg("fg_ari"),
        "miou":   agg("miou"),
        "vehicle_iou_semantic": agg("vehicle_iou_semantic"),
        "per_clip": per_clip,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print()
    for k in ("fg_ari", "miou", "vehicle_iou_semantic"):
        if result[k] is not None:
            print(f"{k:22s}: {result[k]['mean']:.4f} +/- {result[k]['std']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Reconstruction metrics (MSE, SSIM) restricted to foreground pixels.

Motivation. Whole-image MSE/SSIM on BEV scenes is dominated by the road
and background classes (~90+ % of pixels), so small differences in how
the encoder reconstructs actual objects are drowned out. This script
reports MSE and SSIM for three regions:
    - global:  whole 192x192 image
    - fg:      pixels where the GT instance mask (from
               generate_gt_masks.py) is nonzero, i.e. vehicles and
               pedestrians rasterised from actors_data
    - bg:      pixels where the GT instance mask is zero, i.e.
               everything the fg mask excludes (road, sidewalk, sky,
               and any static scene structure not represented in
               actors_data).

Reads:
    - Manifest from prepare_test_manifest.py.
    - GT instance masks from generate_gt_masks.py.
    - SAVi checkpoint.

Writes: <out> with a `per_clip` list of dicts and aggregated
mean/std across clips for each region and each metric. Values are
reported on the [0, 1] pixel-intensity scale so they line up with
the training-time W&B curves.

Example:
    python eval_fg_reconstruction.py \\
        --params slotformer/bev_slots/configs/savi_bev_params.py \\
        --weight $WEIGHT \\
        --manifest manifest.json \\
        --gt-dir gt_masks \\
        --out results_fg_recon.json
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
from skimage.metrics import structural_similarity as ssim_fn
from torchvision import transforms

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "slotformer" / "video_prediction"))

from slotformer.bev_slots.models import build_model  # noqa: E402


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
    return torch.stack(imgs, dim=0)  # [T, C, H, W]


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean of `values` over the pixels where `mask` is True. Returns NaN
    if the mask has no True pixels."""
    m = mask.astype(bool)
    if not m.any():
        return float("nan")
    return float(values[m].mean())


def per_frame_metrics(x: np.ndarray, y: np.ndarray, fg: np.ndarray) -> dict:
    """
    x: [3, H, W] float in [0, 1], target image.
    y: [3, H, W] float in [0, 1], reconstruction.
    fg: [H, W] bool foreground mask.
    """
    per_pixel_sq_err = ((x - y) ** 2).mean(axis=0)  # [H, W]
    mse_global = float(per_pixel_sq_err.mean())
    mse_fg = masked_mean(per_pixel_sq_err, fg)
    mse_bg = masked_mean(per_pixel_sq_err, ~fg)

    # skimage SSIM with `full=True` returns the per-pixel SSIM map.
    # Window and sigma matched to visualize_savi_slots.py's settings.
    _, ssim_map = ssim_fn(
        x, y,
        channel_axis=0, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False, data_range=1.0, full=True,
    )
    # ssim_map has shape [3, H, W] with a valid window inside; average
    # over channels to get a per-pixel score.
    ssim_map = ssim_map.mean(axis=0)  # [H, W]
    ssim_global = float(ssim_map.mean())
    ssim_fg = masked_mean(ssim_map, fg)
    ssim_bg = masked_mean(ssim_map, ~fg)

    return {
        "mse_global": mse_global,
        "mse_fg": mse_fg,
        "mse_bg": mse_bg,
        "ssim_global": ssim_global,
        "ssim_fg": ssim_fg,
        "ssim_bg": ssim_bg,
        "fg_pixel_fraction": float(fg.mean()),
    }


def nan_mean_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(len(arr))}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True, type=str)
    parser.add_argument("--weight", required=True, type=str)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path,
                        help="Instance masks from generate_gt_masks.py.")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Optional override: directory containing "
                             "the route folders. If given, birdview JPGs "
                             "are loaded from <data-root>/<route_name>/birdview/ "
                             "instead of from the absolute path in the manifest. "
                             "Use this when the manifest's route_path points to a "
                             "a stale extraction path.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

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
    model.testing = False  # need reconstructions in the output dict
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(params.resolution),
        transforms.ToTensor(),
    ])

    manifest = json.loads(args.manifest.read_text())
    per_clip = []
    for entry in manifest["entries"]:
        obj3d_id = entry["obj3d_id"]
        if args.data_root is not None:
            route = args.data_root / entry["route_name"]
        else:
            route = Path(entry["route_path"])
        frames = entry["frame_indices"]

        video = load_frames(route, frames, transform).to(device)  # [T, C, H, W]
        out_dict = model({"img": video[None]})
        recon = out_dict["post_recon_combined"][0].cpu().numpy()  # [T, C, H, W]
        target = video.cpu().numpy()                              # [T, C, H, W]
        gt = np.load(args.gt_dir / f"{obj3d_id}_masks.npy")       # [T, H, W]

        per_frame = []
        for t in range(target.shape[0]):
            fg = gt[t] > 0
            per_frame.append(per_frame_metrics(target[t], recon[t], fg))

        # per-clip aggregate: average across the 6 frames of this clip.
        keys = per_frame[0].keys()
        clip_rec = {"obj3d_id": obj3d_id, "route_name": entry["route_name"],
                    "n_frames": len(per_frame)}
        for k in keys:
            vals = [pf[k] for pf in per_frame]
            clip_rec[k] = nan_mean_std(vals)
        per_clip.append(clip_rec)

        # Log a compact line
        def m(k):
            v = clip_rec[k]["mean"]
            return f"{v:.5f}" if v is not None else "  nan"
        print(f"  {obj3d_id:>3s}  "
              f"mse[glob={m('mse_global')} fg={m('mse_fg')} bg={m('mse_bg')}]  "
              f"ssim[glob={m('ssim_global')} fg={m('ssim_fg')} bg={m('ssim_bg')}]  "
              f"fg_pct={clip_rec['fg_pixel_fraction']['mean']*100:.2f}")

    # Cross-clip aggregate: mean and std of the per-clip means.
    def agg(k):
        vals = [c[k]["mean"] for c in per_clip if c[k]["mean"] is not None]
        return nan_mean_std(vals)

    result = {
        "weight": args.weight,
        "params": args.params,
        "manifest": str(args.manifest),
        "gt_dir": str(args.gt_dir),
        "n_clips": len(per_clip),
        "mse_global": agg("mse_global"),
        "mse_fg":     agg("mse_fg"),
        "mse_bg":     agg("mse_bg"),
        "ssim_global": agg("ssim_global"),
        "ssim_fg":     agg("ssim_fg"),
        "ssim_bg":     agg("ssim_bg"),
        "fg_pixel_fraction": agg("fg_pixel_fraction"),
        "per_clip": per_clip,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print()
    for k in ("mse_global", "mse_fg", "mse_bg",
              "ssim_global", "ssim_fg", "ssim_bg"):
        a = result[k]
        if a["mean"] is not None:
            print(f"{k:12s}: {a['mean']:.5f} +/- {a['std']:.5f}  (n={a['n']})")
    fg = result["fg_pixel_fraction"]
    print(f"fg pixel %  : {fg['mean']*100:.2f} +/- {fg['std']*100:.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Generate SEMANTIC ground-truth masks from LMDrive's topdown/*.jpg.

topdown/*.jpg is CARLA's built-in semantic-segmentation camera output. Each
pixel value is a CARLA class ID:
    0=Unlabeled, 1=Building, 2=Fence, 3=Other, 4=Pedestrian, 5=Pole,
    6=RoadLine, 7=Road, 8=Sidewalk, 9=Vegetation, 10=Vehicles, 11=Wall,
    12=TrafficSign, 13=Sky, 14=Ground, 15=Bridge, 16=RailTrack, 17=GuardRail,
    18=TrafficLight, 19=Static, 20=Dynamic, 21=Water, 22=Terrain
(LMDrive additionally overlays traffic-light and stop-sign markers at higher
values via map_agent.py:59-60 — safe to ignore for vehicle-class masks.)

Camera geometry (leaderboard/team_code/map_agent.py:11-27):
    512x512, pitch=-90 deg, FoV=50 deg, mounted 100 m above ego.
    At 100 m altitude with FoV 50 deg, ground coverage is
        2 * 100 * tan(25 deg) ~= 93.26 m per side,
    i.e. ~5.49 px/m. Ego-forward = image-up, ego-centred.

The birdview covers 100 m x 100 m at 4 px/m — a slightly LARGER extent than
topdown's ~93 m. This script centre-crops the birdview footprint out of
topdown by resampling the shared 93 m x 93 m centre region and pads (or
masks) the outer ring where topdown has no coverage. Since the outer ring is
only ~3.4 m per side, this affects a small fraction of pixels.

Output: <out-dir>/<obj3d_id>_semantic.npy — uint8 array [T, 192, 192]
with values matching CARLA semantic class IDs (0-22).

For a *vehicle-only* binary mask, use --vehicle-only which emits 0/1 arrays
where 1 = pixels belonging to class 10.

Example:
    python generate_semantic_masks.py \\
        --manifest manifest.json \\
        --out-dir semantic_masks \\
        --vehicle-only
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

# Topdown camera geometry (see docstring)
TOPDOWN_W = 512
TOPDOWN_H = 512
TOPDOWN_FOV_DEG = 50.0
TOPDOWN_ALT_M = 100.0
TOPDOWN_EXTENT_M = 2 * TOPDOWN_ALT_M * math.tan(math.radians(TOPDOWN_FOV_DEG / 2))
TOPDOWN_PPM = TOPDOWN_W / TOPDOWN_EXTENT_M  # ~5.49

# Birdview geometry (matches generate_gt_masks.py)
BEV_EXTENT_M = 100.0  # 400 px @ 4 ppm
MODEL_H = 192
MODEL_W = 192

VEHICLE_CLASS_ID = 10


def load_topdown(route: Path, frame: int) -> np.ndarray:
    img = Image.open(route / "topdown" / f"{frame:04d}.jpg").convert("L")
    return np.asarray(img, dtype=np.uint8)  # [H, W]


def align_topdown_to_birdview_extent(top: np.ndarray) -> np.ndarray:
    """Resize the 512x512 topdown (93 m extent) to a canvas matching the
    birdview's 100 m extent, then centre-place. Uncovered outer ring is left
    at 0 (semantic Unlabeled).

    Returns a [target_h, target_w] uint8 array where the centre region carries
    topdown pixels and the outer ring is 0.
    """
    # Match birdview extent (100 m) at the model's target ppm (MODEL_W / 100)
    target_ppm = MODEL_W / BEV_EXTENT_M  # 1.92 px/m
    # topdown covers ~93.26 m -> resample to (93.26 * target_ppm) pixels
    top_size_at_target = int(round(TOPDOWN_EXTENT_M * target_ppm))
    top_img = Image.fromarray(top, mode="L")
    top_resized = np.asarray(
        top_img.resize((top_size_at_target, top_size_at_target), Image.NEAREST),
        dtype=np.uint8,
    )
    canvas = np.zeros((MODEL_H, MODEL_W), dtype=np.uint8)
    off = (MODEL_H - top_size_at_target) // 2
    end = off + top_size_at_target
    if off < 0:
        # topdown ppm larger than target ppm: centre-crop it
        crop = (-off, -off + MODEL_H)
        canvas[:] = top_resized[crop[0]:crop[1], crop[0]:crop[1]]
    else:
        canvas[off:end, off:end] = top_resized
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--vehicle-only", action="store_true",
                        help="Emit binary masks (1 = vehicle class, 0 else).")
    parser.add_argument("--debug-dir", type=Path, default=None,
                        help="Save per-frame overlays of topdown-aligned mask on birdview.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest["entries"]:
        route = Path(entry["route_path"])
        obj3d_id = entry["obj3d_id"]
        frames = entry["frame_indices"]
        clip = []
        for t, f in enumerate(frames):
            top = load_topdown(route, f)
            aligned = align_topdown_to_birdview_extent(top)
            if args.vehicle_only:
                aligned = (aligned == VEHICLE_CLASS_ID).astype(np.uint8)
            clip.append(aligned)

            if args.debug_dir is not None:
                # Overlay aligned semantic mask on the (resized) birdview
                bv = Image.open(route / "birdview" / f"{f:04d}.jpg").convert("RGB")
                bv = bv.resize((MODEL_W, MODEL_H), Image.BILINEAR)
                rng = np.random.default_rng(0)
                n = int(aligned.max()) + 1
                palette = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
                palette[0] = 0
                coloured = Image.fromarray(palette[aligned], mode="RGB")
                Image.blend(bv, coloured, alpha=0.5).save(
                    args.debug_dir / f"{obj3d_id}_frame{t:02d}.png"
                )
        clip = np.stack(clip, axis=0)
        np.save(args.out_dir / f"{obj3d_id}_semantic.npy", clip)
        suffix = " (vehicle-only binary)" if args.vehicle_only else ""
        print(f"  {obj3d_id}_semantic.npy shape={clip.shape} classes={sorted(int(v) for v in np.unique(clip))}{suffix}")


if __name__ == "__main__":
    main()

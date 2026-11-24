"""Rasterise per-frame ground-truth INSTANCE masks from LMDrive actor metadata.

For each (video, frame) pair in the manifest:
    1. Load actors_data/{frame:04d}.json (per-actor loc/ori/box).
    2. Load measurements/{frame:04d}.json (ego x, y, theta).
    3. Skip the ego actor (identified by position match).
    4. Project each actor's rotated rectangle into the ego-relative BEV frame.
    5. Rasterise with a unique instance ID.
    6. Nearest-neighbour resize to the model's 192x192 input resolution.

Projection convention (verified against LMDrive source and sample data — no
guessing):
    - LMDrive stores `theta = compass` in radians, defined as radians(90 + yaw_deg)
      (see leaderboard/team_code/lmdriver_agent.py:248 and map_agent.py:73).
    - Ego's world-space forward unit vector is (sin(theta), -cos(theta)).
      Verified: at frame 10 of the example route, theta = 0.0006 rad and the
      ego's own actors_data ori vector is (0.0006, -1.0) — matches exactly.
    - Ego's world-space right unit vector is (-cos(theta), -sin(theta)) — this
      is rot90CW(forward) in CARLA's left-handed frame.
    - BEV geometry: 400x400 @ 4 px/m, ego at center (200, 200), ego-forward = image-up.
      Matches carla_birdeye_view.BirdViewProducer with target_size=400x400,
      pixels_per_meter=4, crop_type=FRONT_AND_REAR_AREA (see
      leaderboard/team_code/lmdriver_agent.py:159-161).

Outputs:
    <out-dir>/<obj3d_id>_masks.npy  — int32 array [T, 192, 192]
      (0 = background, otherwise a per-clip-unique instance ID)

Optionally, with --debug-dir, per-frame overlay PNGs blending the birdview
with the coloured instance mask so alignment is verifiable at a glance.

Example:
    python generate_gt_masks.py \\
        --manifest manifest.json \\
        --out-dir gt_masks \\
        --debug-dir gt_debug
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# BEV geometry — matches leaderboard/team_code/lmdriver_agent.py:159-161
BEV_W = 400
BEV_H = 400
PIXELS_PER_METER = 4
CX = BEV_W / 2.0
CY = BEV_H / 2.0

# Model input resolution (vision_encoder/timm/data/carla_dataset.py:178-180)
MODEL_H = 192
MODEL_W = 192

# Ego identification: an actor whose position lies within this distance of the
# ego's own measurement is treated as the ego and skipped.
EGO_MATCH_TOL_M = 0.5


def load_actors(route: Path, frame: int) -> dict:
    with open(route / "actors_data" / f"{frame:04d}.json") as f:
        return json.load(f)


def load_ego(route: Path, frame: int) -> tuple:
    with open(route / "measurements" / f"{frame:04d}.json") as f:
        m = json.load(f)
    return float(m["x"]), float(m["y"]), float(m["theta"])


def actor_corners_world(loc, ori, box_half):
    """Return the four (wx, wy) corners of an actor's ground-plane rectangle.

    Uses the actor's own `ori` unit vector directly (no yaw re-derivation), so
    the result is exact regardless of any yaw-sign convention questions.

    loc = [x, y, z] world position (metres).
    ori = [ox, oy, oz] unit vector in world frame pointing forward.
    box_half = [half_length_x, half_width_y] in the actor's local frame.
    """
    x, y = loc[0], loc[1]
    ox, oy = ori[0], ori[1]
    hl, hw = box_half[0], box_half[1]
    # local frame: +x = forward, +y = right; corners: BR, BL, FL, FR
    local = [(-hl, -hw), (-hl, +hw), (+hl, +hw), (+hl, -hw)]
    corners = []
    for lx, ly in local:
        # world offset = forward*ori + right*rot90CW(ori)
        wx = x + lx * ox + ly * oy
        wy = y + lx * oy - ly * ox
        corners.append((wx, wy))
    return corners


def world_to_bev_pixel(wx, wy, ex, ey, etheta):
    """CARLA world (wx, wy) -> BEV image (u, v).

    Ego at world (ex, ey), heading etheta radians (LMDrive compass convention).
    Ego forward vector: (sin(theta), -cos(theta)).
    Ego right   vector: (-cos(theta), -sin(theta))  [= rot90CW of forward].
    """
    dx = wx - ex
    dy = wy - ey
    sin_t = math.sin(etheta)
    cos_t = math.cos(etheta)
    forward = dx * sin_t + dy * (-cos_t)
    right = dx * (-cos_t) + dy * (-sin_t)
    u = CX + right * PIXELS_PER_METER
    v = CY - forward * PIXELS_PER_METER
    return u, v


def is_ego_actor(actor_loc, ex, ey) -> bool:
    dx, dy = actor_loc[0] - ex, actor_loc[1] - ey
    return (dx * dx + dy * dy) < (EGO_MATCH_TOL_M * EGO_MATCH_TOL_M)


def rasterise_frame(route: Path, frame: int) -> tuple:
    """Return (mask, stats). Mask is int32 [BEV_H, BEV_W] with unique
    instance IDs (0 = background). stats is a dict with counters."""
    actors = load_actors(route, frame)
    ex, ey, etheta = load_ego(route, frame)

    mask = Image.new("I", (BEV_W, BEV_H), 0)
    draw = ImageDraw.Draw(mask)
    inst_id = 0
    n_total = len(actors)
    n_ego = n_missing = n_offscreen = 0
    for actor_id, a in actors.items():
        # Defensive: some actor entries (traffic lights, walkers with a
        # different schema, sensors) may lack loc/ori/box. Skip them.
        loc = a.get("loc")
        ori = a.get("ori")
        box = a.get("box")
        if loc is None or ori is None or box is None:
            n_missing += 1
            continue
        if is_ego_actor(loc, ex, ey):
            n_ego += 1
            continue
        corners_w = actor_corners_world(loc, ori, box)
        corners_px = [world_to_bev_pixel(wx, wy, ex, ey, etheta)
                      for (wx, wy) in corners_w]
        # Skip actors entirely outside the BEV FoV
        if all((u < 0 or u >= BEV_W or v < 0 or v >= BEV_H)
               for (u, v) in corners_px):
            n_offscreen += 1
            continue
        inst_id += 1
        draw.polygon(corners_px, fill=inst_id)
    stats = {
        "n_actors": n_total,
        "n_drawn": inst_id,
        "n_skipped_ego": n_ego,
        "n_skipped_missing_fields": n_missing,
        "n_skipped_offscreen": n_offscreen,
    }
    return np.asarray(mask, dtype=np.int32), stats


def resize_mask_nn(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    # PIL mode "I" is 32-bit signed integer; use int32 for round-trip fidelity.
    img = Image.fromarray(mask.astype(np.int32), mode="I")
    img = img.resize((out_w, out_h), resample=Image.NEAREST)
    return np.asarray(img, dtype=np.int32)


def render_debug_overlay(route: Path, frame: int, mask: np.ndarray) -> Image.Image:
    bv = Image.open(route / "birdview" / f"{frame:04d}.jpg").convert("RGB")
    if bv.size != (BEV_W, BEV_H):
        bv = bv.resize((BEV_W, BEV_H), resample=Image.BILINEAR)
    rng = np.random.default_rng(0)
    n = int(mask.max()) + 1
    palette = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    palette[0] = 0
    coloured = palette[mask]
    overlay = Image.fromarray(coloured, mode="RGB")
    return Image.blend(bv, overlay, alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--debug-dir", type=Path, default=None,
                        help="If given, save per-frame overlay PNGs here.")
    parser.add_argument("--no-resize", action="store_true",
                        help="Keep masks at 400x400 (default resizes to 192x192).")
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
        clip_stats = []
        for t, f in enumerate(frames):
            mask_400, stats = rasterise_frame(route, f)
            clip_stats.append(stats)
            out_mask = mask_400 if args.no_resize else resize_mask_nn(mask_400, MODEL_H, MODEL_W)
            clip.append(out_mask)
            if args.debug_dir is not None:
                render_debug_overlay(route, f, mask_400).save(
                    args.debug_dir / f"{obj3d_id}_frame{t:02d}.png"
                )
        clip = np.stack(clip, axis=0)
        np.save(args.out_dir / f"{obj3d_id}_masks.npy", clip)
        avg = lambda k: sum(s[k] for s in clip_stats) / len(clip_stats)
        print(f"  {obj3d_id}_masks.npy shape={clip.shape} "
              f"n_instances_first={int(clip[0].max())} "
              f"avg drawn/actors={avg('n_drawn'):.1f}/{avg('n_actors'):.1f} "
              f"skipped(missing_fields={avg('n_skipped_missing_fields'):.1f}, "
              f"ego={avg('n_skipped_ego'):.1f}, "
              f"offscreen={avg('n_skipped_offscreen'):.1f})")


if __name__ == "__main__":
    main()

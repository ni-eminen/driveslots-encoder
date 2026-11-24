"""Reproduce the exact 72-frame selection used by visualize_savi_slots.py.

This script does NOT move any files (unlike scripts/data_preproc/create_obj3d_style_dataset.py).
It walks the extracted test_town shard, replays the same test-split logic, and
records for each of the 12 selected videos:
    - the numeric OBJ3D-style ID it would have received
    - the ORIGINAL route directory (so actors_data/, measurements/, birdview/
      remain accessible for GT mask generation)
    - the 6 frame indices that load_video_frames would pick under seed=42

Emits a JSON manifest suitable for consumption by generate_gt_masks.py and
eval_fg_ari_miou.py.

Example:
    python prepare_test_manifest.py \\
        --data-root $EXTRACT_DIR \\
        --out manifest.json \\
        --n-samples 12 --n-frames 6 --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path


def collect_birdview_folders(root: Path):
    """Exact analogue of create_obj3d_style_dataset.py:42."""
    return sorted([p for p in root.glob("*/birdview") if p.is_dir()], key=str)


def compute_splits(n: int):
    """Exact analogue of create_obj3d_style_dataset.py:47-48."""
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.10))
    return n_test, n_val


def load_video_frame_starts(folders_by_test_id, n_frames, frame_offset, min_start, seed):
    """Replay load_video_frames' random.randint calls in the same order.

    visualize_savi_slots.py calls random.seed(seed) once, then for each
    selected video (in linspace-index order) calls random.randint once via
    load_video_frames. Reproducing that sequence gives the exact start indices.
    """
    random.seed(seed)
    starts = []
    for folder in folders_by_test_id:
        num_files = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
        max_start = num_files - (n_frames - 1) * frame_offset - 1
        start = random.randint(min(min_start, max_start), max_start)
        starts.append(start)
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", required=True, type=Path,
        help="Extracted test_town directory (the parent that contains */birdview).",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--n-frames", type=int, default=6)
    parser.add_argument("--frame-offset", type=int, default=1)
    parser.add_argument("--min-start", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    birdviews = collect_birdview_folders(args.data_root)
    n = len(birdviews)
    if n == 0:
        raise SystemExit(f"No */birdview folders under {args.data_root}")

    n_test, n_val = compute_splits(n)
    test_folders_by_original_order = birdviews[:n_test]

    # create_obj3d_style_dataset.py renumbers test folders as 0, 1, ..., n_test-1
    # in the order they were encountered. visualize_savi_slots.py then does
    # glob_all(data_root/test, only_dir=True) and sorted() — a STRING sort on
    # the numeric folder names. String sort of "0", "1", ..., "n_test-1" gives
    # "0", "1", "10", "100", ..., "11", ..., "2", "20", ... which does NOT
    # match numeric order once n_test >= 10. We reproduce that string sort.
    numeric_ids = [str(i) for i in range(n_test)]
    string_sorted_ids = sorted(numeric_ids)
    id_to_route = {str(i): test_folders_by_original_order[i].parent
                   for i in range(n_test)}

    # Uniform linspace(0, n_test - 1, n_samples) after string sort — like torch.linspace(...).long()
    n_samples = min(args.n_samples, n_test)
    if n_samples == 1:
        picks = [0]
    else:
        picks = [int(round(i * (n_test - 1) / (n_samples - 1))) for i in range(n_samples)]
    selected_ids = [string_sorted_ids[i] for i in picks]
    selected_routes = [id_to_route[i] for i in selected_ids]
    selected_birdviews = [r / "birdview" for r in selected_routes]

    starts = load_video_frame_starts(
        selected_birdviews,
        n_frames=args.n_frames,
        frame_offset=args.frame_offset,
        min_start=args.min_start,
        seed=args.seed,
    )

    entries = []
    for obj3d_id, route, start in zip(selected_ids, selected_routes, starts):
        frame_indices = [start + n * args.frame_offset for n in range(args.n_frames)]
        entries.append({
            "obj3d_id": obj3d_id,
            "route_path": str(route),
            "route_name": route.name,
            "frame_indices": frame_indices,
        })

    manifest = {
        "data_root": str(args.data_root),
        "n_test_videos": n_test,
        "n_val_videos": n_val,
        "n_total_videos": n,
        "n_samples": n_samples,
        "n_frames": args.n_frames,
        "frame_offset": args.frame_offset,
        "min_start": args.min_start,
        "seed": args.seed,
        "entries": entries,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out} ({len(entries)} entries)")
    print(f"first 3: {[(e['obj3d_id'], e['route_name'], e['frame_indices'][0]) for e in entries[:3]]}")


if __name__ == "__main__":
    main()

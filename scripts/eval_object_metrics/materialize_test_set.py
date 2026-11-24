"""Copy the 72 held-out frames into a persistent test set.

Reads a manifest produced by prepare_test_manifest.py (whose route_paths
point at an extracted test dataset) and copies only
the referenced files --- birdview/, actors_data/, measurements/, and
topdown/ for the 6 frames of each of the 12 videos --- into a target
directory. Rewrites the manifest's paths so future evaluations can
consume the persistent copy directly with no re-extraction.

Output layout:
    <target>/
        manifest.json                # rewritten to reference <target>/routes/...
        routes/
            <route_name>/
                birdview/{f:04d}.jpg          for each frame
                actors_data/{f:04d}.json      for each frame
                measurements/{f:04d}.json     for each frame
                topdown/{f:04d}.jpg           for each frame

The rewritten manifest carries both `route_path` (absolute, pointing at
`<target>/routes/<route_name>`) and `route_name`, so scripts that use
either continue to work.

Example:
    python materialize_test_set.py \\
        --manifest $OUT_ROOT/manifest.json \\
        --target /path/to/data
"""

import argparse
import json
import shutil
from pathlib import Path

SUBDIRS = {
    "birdview":     "{f:04d}.jpg",
    "actors_data":  "{f:04d}.json",
    "measurements": "{f:04d}.json",
    "topdown":      "{f:04d}.jpg",
}


def copy_frame_files(src_route: Path, dst_route: Path, frames):
    """Copy the specific per-frame files listed in `frames` for each
    of the required subfolders. Silently skips subfolders that don't
    exist in the source (e.g., a route missing topdown/)."""
    n_copied = 0
    for sub, fname_fmt in SUBDIRS.items():
        src_sub = src_route / sub
        if not src_sub.is_dir():
            print(f"    WARN: {src_sub} missing, skipping")
            continue
        dst_sub = dst_route / sub
        dst_sub.mkdir(parents=True, exist_ok=True)
        for f in frames:
            src_file = src_sub / fname_fmt.format(f=f)
            if not src_file.is_file():
                print(f"    WARN: {src_file} missing")
                continue
            shutil.copy2(src_file, dst_sub / src_file.name)
            n_copied += 1
    return n_copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Manifest from prepare_test_manifest.py.")
    parser.add_argument("--target", required=True, type=Path,
                        help="Persistent target directory.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    args.target.mkdir(parents=True, exist_ok=True)
    routes_root = args.target / "routes"
    routes_root.mkdir(exist_ok=True)

    new_entries = []
    for entry in manifest["entries"]:
        route_name = entry["route_name"]
        src_route = Path(entry["route_path"])
        dst_route = routes_root / route_name
        dst_route.mkdir(exist_ok=True)
        n = copy_frame_files(src_route, dst_route, entry["frame_indices"])
        print(f"  {entry['obj3d_id']:>3s}  {route_name}  copied {n} files")
        new_entries.append({
            **entry,
            "route_path": str(dst_route.resolve()),
        })

    new_manifest = {
        **manifest,
        "data_root": str(routes_root.resolve()),
        "source_data_root": manifest.get("data_root"),
        "source_manifest": str(args.manifest.resolve()),
        "entries": new_entries,
    }
    out_manifest = args.target / "manifest.json"
    out_manifest.write_text(json.dumps(new_manifest, indent=2))
    print()
    print(f"wrote {out_manifest}")
    print(f"target size:")
    import subprocess
    subprocess.run(["du", "-sh", str(args.target)])


if __name__ == "__main__":
    main()

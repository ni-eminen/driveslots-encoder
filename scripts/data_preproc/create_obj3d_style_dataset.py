import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


TOWNS = {1, 2, 3, 4, 5, 6, 7, 10}

TEST_WEATHERS = {12, 13, 20}

TRAIN_VAL_WEATHERS = (
    set(range(0, 12))
    | set(range(14, 20))
)

ROUTE_RE = re.compile(
    r"^routes_town(\d+)_.*?_w(\d+)_"
)


def parse_route(route_name: str):
    match = ROUTE_RE.match(route_name)

    if match is None:
        return None

    town = int(match.group(1))
    weather = int(match.group(2))

    return town, weather


def build_split(
    root: Path,
    val_fraction: float,
    seed: int,
):
    test_routes = []
    train_val_groups = defaultdict(list)

    birdviews = sorted(
        (
            p
            for p in root.glob("*/birdview")
            if p.is_dir()
        ),
        key=str,
    )

    for bv in birdviews:
        route_name = bv.parent.name
        parsed = parse_route(route_name)

        if parsed is None:
            continue

        town, weather = parsed

        if town not in TOWNS:
            continue

        if weather in TEST_WEATHERS:
            test_routes.append(bv)

        elif weather in TRAIN_VAL_WEATHERS:
            train_val_groups[(town, weather)].append(bv)

    train_routes = []
    val_routes = []

    # Stratified deterministic validation split.
    for (town, weather), routes in sorted(train_val_groups.items()):
        routes = sorted(routes, key=str)

        rng = random.Random(
            seed + town * 1000 + weather
        )
        rng.shuffle(routes)

        if len(routes) <= 1:
            # Keep the only route in training.
            n_val = 0
        else:
            n_val = round(len(routes) * val_fraction)

            # Ensure validation is represented but don't
            # consume an entire group.
            n_val = max(1, n_val)
            n_val = min(n_val, len(routes) - 1)

        val_routes.extend(routes[:n_val])
        train_routes.extend(routes[n_val:])

    return {
        "train": sorted(train_routes, key=str),
        "val": sorted(val_routes, key=str),
        "test": sorted(test_routes, key=str),
    }


def move_routes(
    splits,
    out_root: Path,
):
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = out_root / "split_manifest.tsv"

    with manifest.open("w") as f:
        f.write("split\tid\troute\ttown\tweather\n")

        for split in ("train", "val", "test"):
            routes = splits[split]

            split_root = out_root / split
            split_root.mkdir(parents=True, exist_ok=True)

            print(f"{split}: {len(routes)} routes")

            for sid, bv in enumerate(routes):
                route_name = bv.parent.name
                town, weather = parse_route(route_name)

                dst = split_root / str(sid)

                if dst.exists():
                    raise RuntimeError(
                        f"Destination already exists: {dst}"
                    )

                # Move each source sequence into the split layout.
                shutil.move(str(bv), str(dst))

                f.write(
                    f"{split}\t{sid}\t{route_name}"
                    f"\t{town}\t{weather}\n"
                )

                if (sid + 1) % 100 == 0:
                    print(
                        f"  {split}: {sid + 1}/{len(routes)}"
                    )


def print_summary(out_root: Path):
    print()
    print("Final structure:")
    print(out_root)

    for split in ("train", "val", "test"):
        split_root = out_root / split

        n_routes = sum(
            1
            for p in split_root.iterdir()
            if p.is_dir()
        )

        print(f"  {split}: {n_routes} routes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_root",
        type=Path,
    )

    parser.add_argument(
        "output_root",
        type=Path,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    root = args.input_root.resolve()
    out_root = args.output_root.resolve()

    splits = build_split(
        root=root,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    if not any(splits.values()):
        raise RuntimeError(
            f"No matching routes found under {root}"
        )

    print("Split:")
    print(f"  train: {len(splits['train'])}")
    print(f"  val:   {len(splits['val'])}")
    print(f"  test:  {len(splits['test'])}")

    move_routes(
        splits=splits,
        out_root=out_root,
    )

    print_summary(out_root)
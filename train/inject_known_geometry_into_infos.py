from __future__ import annotations

import argparse
import pickle
from copy import deepcopy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inject explicit known geometry fields into MMDetection3D mono "
            "KITTI info files for reduced-DoF evaluation."
        ))
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source info pickle path.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output info pickle path.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.")
    return parser.parse_args()


def _extract_single_instance(item: dict) -> dict:
    instances = item.get("instances", [])
    if len(instances) != 1:
        raise ValueError(
            f"Expected exactly one instance for sample {item.get('sample_idx')}, "
            f"but found {len(instances)}.")
    instance = instances[0]
    bbox_3d = instance.get("bbox_3d")
    if bbox_3d is None or len(bbox_3d) < 6:
        raise ValueError(
            f"Missing valid bbox_3d for sample {item.get('sample_idx')}.")
    return instance


def _inject_known_geometry(item: dict) -> dict:
    instance = _extract_single_instance(item)
    x, y_bottom, z, length, height, width, *rest = instance["bbox_3d"]
    gravity_y = float(y_bottom) - float(height) * 0.5

    updated = deepcopy(item)
    updated["known_dims"] = [float(length), float(height), float(width)]
    updated["known_gravity_y"] = gravity_y
    return updated


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --force to overwrite.")

    with args.input.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        data_list = payload["data_list"]
        payload = dict(payload)
        payload["data_list"] = [_inject_known_geometry(item) for item in data_list]
    else:
        payload = [_inject_known_geometry(item) for item in payload]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(payload, handle)

    print(f"[done] wrote explicit-known-geometry infos to {args.output}")


if __name__ == "__main__":
    main()

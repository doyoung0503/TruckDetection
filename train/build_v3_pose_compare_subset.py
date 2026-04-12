#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a minimal raw-v3 subset containing only the labels/images "
            "needed for KITTI pose convention comparison on another machine."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Raw v3 root")
    parser.add_argument("--output-root", type=Path, required=True, help="Subset output root")
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=["000000", "000007", "000008", "000043", "000120"],
        help="Sample ids to include",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output root if it exists")
    return parser.parse_args()


def _resolve_source_label(source_root: Path, sample_id: str) -> Path:
    numeric_id = int(sample_id)
    candidates = [
        source_root / "labels" / f"label_{sample_id}.json",
        source_root / "labels" / f"label_{numeric_id:04d}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No source label found for sample {sample_id}: {candidates}")


def _resolve_source_image(source_root: Path, sample_id: str) -> Path:
    numeric_id = int(sample_id)
    candidates = [
        source_root / "images" / f"image_{sample_id}.png",
        source_root / "images" / f"image_{numeric_id:04d}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No source image found for sample {sample_id}: {candidates}")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} already exists. Use --force to overwrite.")
        shutil.rmtree(output_root)

    (output_root / "labels").mkdir(parents=True, exist_ok=True)
    (output_root / "images").mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "sample_ids": args.sample_ids,
        "files": [],
    }

    for sample_id in args.sample_ids:
        src_label = _resolve_source_label(source_root, sample_id)
        src_image = _resolve_source_image(source_root, sample_id)
        dst_label = output_root / "labels" / src_label.name
        dst_image = output_root / "images" / src_image.name
        shutil.copy2(src_label, dst_label)
        shutil.copy2(src_image, dst_image)
        manifest["files"].append(
            {
                "sample_id": sample_id,
                "label": dst_label.name,
                "image": dst_image.name,
            }
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

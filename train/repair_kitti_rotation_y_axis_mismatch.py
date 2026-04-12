from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import export_v3_to_kitti_letterbox as exporter


def normalize_angle_rad(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def patch_line(line: str) -> str:
    fields = line.strip().split()
    if len(fields) < 15:
        raise ValueError(f"Expected at least 15 KITTI fields, got {len(fields)}: {line!r}")

    x = float(fields[11])
    z = float(fields[13])
    ry = float(fields[14])

    new_ry = normalize_angle_rad(ry + math.pi / 2.0)
    new_alpha = normalize_angle_rad(new_ry - math.atan2(x, z + 1e-7))

    fields[3] = f"{new_alpha:.6f}"
    fields[14] = f"{new_ry:.6f}"
    return " ".join(fields)


def read_p2(path: Path) -> np.ndarray:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("P2:"):
            vals = np.array([float(v) for v in line.split()[1:]], dtype=np.float32)
            return vals.reshape(3, 4)[:, :3]
    raise ValueError(f"P2 not found in {path}")


def refine_line(line: str, calib_path: Path) -> str:
    fields = line.strip().split()
    if len(fields) < 15:
        raise ValueError(f"Expected at least 15 KITTI fields, got {len(fields)}: {line!r}")

    bbox = np.array([float(v) for v in fields[4:8]], dtype=np.float32)
    h, w, l = [float(v) for v in fields[8:11]]
    x, y, z = [float(v) for v in fields[11:14]]
    ry = float(fields[14])
    k3 = read_p2(calib_path)

    refined_loc, refined_ry, _ = exporter.refine_pose_to_bbox(
        k3=k3,
        bbox_xyxy=bbox,
        dims_lhw=np.array([l, h, w], dtype=np.float32),
        loc_xyz=np.array([x, y, z], dtype=np.float32),
        ry_init=ry,
        out_w=1280,
        out_h=384,
    )
    refined_x = float(refined_loc[0])
    refined_y = float(refined_loc[1])
    refined_z = float(refined_loc[2])
    refined_alpha = normalize_angle_rad(refined_ry - math.atan2(refined_x, refined_z + 1e-7))
    fields[3] = f"{refined_alpha:.6f}"
    fields[11] = f"{refined_x:.6f}"
    fields[12] = f"{refined_y:.6f}"
    fields[13] = f"{refined_z:.6f}"
    fields[14] = f"{refined_ry:.6f}"
    return " ".join(fields)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Patch an already converted KITTI label_2 directory by rotating "
            "rotation_y by +90 degrees and recomputing alpha. This is a local "
            "repair path for existing exports created before the axis fix."
        )
    )
    p.add_argument("--dataset-root", type=Path, required=True, help="Converted KITTI root.")
    p.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Optional backup directory for the original label_2 tree. Defaults to label_2_backup_pre_rotfix.",
    )
    p.add_argument(
        "--refine-to-bbox",
        action="store_true",
        help="After the +90deg fix, jointly refine x/z/rotation_y per-sample to better match the exported 2D bbox.",
    )
    p.add_argument(
        "--skip-rotfix",
        action="store_true",
        help="Do not apply the legacy +90deg rotation patch; only run the bbox-based pose refinement.",
    )
    p.add_argument("--dry-run", action="store_true", help="Inspect only, do not rewrite files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    label_dir = dataset_root / "training" / "label_2"
    if not label_dir.exists():
        raise FileNotFoundError(f"label_2 not found: {label_dir}")

    files = sorted(label_dir.glob("*.txt"))
    if not files:
        raise RuntimeError(f"No label files found under {label_dir}")

    if not args.dry_run:
        backup_dir = (
            args.backup_dir.resolve()
            if args.backup_dir is not None
            else dataset_root / "training" / "label_2_backup_pre_rotfix"
        )
        if backup_dir.exists():
            print(f"[backup] reuse existing backup: {backup_dir}")
        else:
            shutil.copytree(label_dir, backup_dir)
            print(f"[backup] saved original labels to {backup_dir}")

    changed = 0
    sample_preview: list[tuple[str, float, float, float, float, float, float]] = []
    for path in files:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            continue

        patched_lines = []
        for line in lines:
            fields = line.strip().split()
            old_ry = float(fields[14])
            old_x = float(fields[11])
            old_z = float(fields[13])
            patched = line if args.skip_rotfix else patch_line(line)
            if args.refine_to_bbox:
                patched = refine_line(patched, dataset_root / "training" / "calib" / path.name)
            new_fields = patched.split()
            new_x = float(new_fields[11])
            new_z = float(new_fields[13])
            new_ry = float(new_fields[14])
            patched_lines.append(patched)
            if len(sample_preview) < 5:
                sample_preview.append((path.name, old_x, new_x, old_z, new_z, old_ry, new_ry))
        changed += 1

        if not args.dry_run:
            path.write_text("\n".join(patched_lines) + "\n", encoding="utf-8")

    print(f"[done] processed {changed} files under {label_dir}")
    for name, old_x, new_x, old_z, new_z, old_ry, new_ry in sample_preview:
        print(
            f"[sample] {name}: "
            f"x {old_x:.6f} -> {new_x:.6f} | "
            f"z {old_z:.6f} -> {new_z:.6f} | "
            f"ry {old_ry:.6f} -> {new_ry:.6f}"
        )


if __name__ == "__main__":
    main()

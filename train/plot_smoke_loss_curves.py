from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOSS_RE = re.compile(
    r"iter:\s*(?P<iter>\d+)"
    r".*?loss:\s*(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\([^)]*\)"
    r".*?hm_loss:\s*(?P<hm>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\([^)]*\)"
    r".*?reg_loss:\s*(?P<reg>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\([^)]*\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SMOKE loss curves from training log.txt.")
    parser.add_argument("--log", type=Path, required=True, help="Path to SMOKE log.txt")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save csv and png")
    parser.add_argument("--bucket", type=int, default=100, help="Iteration bucket size")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title")
    return parser.parse_args()


def parse_log(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        m = LOSS_RE.search(line)
        if not m:
            continue
        rows.append(
            {
                "iter": int(m.group("iter")),
                "loss": float(m.group("loss")),
                "hm_loss": float(m.group("hm")),
                "reg_loss": float(m.group("reg")),
            }
        )
    return rows


def bucketize(rows: list[dict[str, float]], bucket: int) -> list[dict[str, float]]:
    buckets: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        bucket_end = ((int(row["iter"]) - 1) // bucket + 1) * bucket
        buckets.setdefault(bucket_end, []).append(row)

    summary: list[dict[str, float]] = []
    for bucket_end in sorted(buckets):
        chunk = buckets[bucket_end]
        summary.append(
            {
                "bucket_end": bucket_end,
                "num_points": len(chunk),
                "loss_mean": float(np.mean([item["loss"] for item in chunk])),
                "hm_loss_mean": float(np.mean([item["hm_loss"] for item in chunk])),
                "reg_loss_mean": float(np.mean([item["reg_loss"] for item in chunk])),
            }
        )
    return summary


def write_csv(path: Path, summary: list[dict[str, float]]) -> None:
    fieldnames = ["bucket_end", "num_points", "loss_mean", "hm_loss_mean", "reg_loss_mean"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def plot_summary(summary: list[dict[str, float]], out_path: Path, title: str) -> None:
    xs = [row["bucket_end"] for row in summary]
    total = [row["loss_mean"] for row in summary]
    hm = [row["hm_loss_mean"] for row in summary]
    reg = [row["reg_loss_mean"] for row in summary]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(xs, total, marker="o", linewidth=2.2, color="#355070", label="Total Loss")
    ax.plot(xs, hm, marker="o", linewidth=2.2, color="#6d597a", label="HM Loss")
    ax.plot(xs, reg, marker="o", linewidth=2.2, color="#e56b6f", label="Reg Loss")
    ax.set_xlabel("Iteration (bucket end)")
    ax.set_ylabel("100-iter Mean Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = parse_log(args.log)
    if not rows:
        raise RuntimeError(f"No loss rows parsed from {args.log}")
    summary = bucketize(rows, args.bucket)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.log.parent.name or args.log.stem
    csv_path = args.output_dir / f"{stem}_loss_bucket{args.bucket}.csv"
    png_path = args.output_dir / f"{stem}_loss_bucket{args.bucket}.png"
    write_csv(csv_path, summary)
    title = args.title or f"{stem} Loss Curves ({args.bucket}-iter mean)"
    plot_summary(summary, png_path, title)

    print(csv_path)
    print(png_path)
    print(f"parsed_points={len(rows)} buckets={len(summary)}")


if __name__ == "__main__":
    main()

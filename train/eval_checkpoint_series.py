"""
Evaluate a series of saved checkpoints after training.

Designed for low-bottleneck training:
  - train first with `--eval-every 0 --save-every 5`
  - run this script afterwards to recover validation metrics per checkpoint
  - merge the results back into the training history JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train.dataset import make_dataloaders
from train.models import build_smoke_model
from train.smoke_loss import build_smoke_loss
from train.smoke_trainer import DATASET_ROOT, RESULTS_DIR, _val_epoch


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _checkpoint_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("epoch_"):
        return int(stem.split("_")[1]), stem
    if stem == "last":
        return 10**9, stem
    if stem == "best":
        return 10**9 + 1, stem
    return 10**9 + 2, stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved checkpoint series on the val split.")
    p.add_argument("--type", required=True, choices=["baseline", "geometry", "baseline_depth", "geometry_aux"])
    p.add_argument("--dataset-root", default=DATASET_ROOT, help="Converted KITTI dataset root")
    p.add_argument("--results-dir", default=str(RESULTS_DIR), help="Training results root")
    p.add_argument("--device", default=_get_device(), help="Evaluation device")
    p.add_argument("--batch", type=int, default=8, help="Validation batch size")
    p.add_argument("--workers", type=int, default=4, help="Validation dataloader workers")
    p.add_argument("--checkpoint-glob", default="epoch_*.pt", help="Checkpoint glob inside model results dir")
    p.add_argument("--include-last", action="store_true", help="Also evaluate last.pt if present")
    p.add_argument("--include-best", action="store_true", help="Also evaluate best.pt if present")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    ckpt_dir = results_dir / args.type
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    checkpoints = list(ckpt_dir.glob(args.checkpoint_glob))
    if args.include_last:
        last_ckpt = ckpt_dir / "last.pt"
        if last_ckpt.exists():
            checkpoints.append(last_ckpt)
    if args.include_best:
        best_ckpt = ckpt_dir / "best.pt"
        if best_ckpt.exists():
            checkpoints.append(best_ckpt)

    checkpoints = sorted({p.resolve() for p in checkpoints}, key=_checkpoint_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    _, val_loader = make_dataloaders(
        root=args.dataset_root,
        model_type=args.type,
        batch_size=args.batch,
        num_workers=args.workers,
        augment=False,
    )

    model = build_smoke_model(args.type, pretrained=False).to(args.device)
    loss_fn = build_smoke_loss(args.type).to(args.device)

    history_path = results_dir / f"history_{args.type}.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = []

    history_by_epoch = {int(h["epoch"]): h for h in history if "epoch" in h}
    eval_rows: list[dict] = []

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model"], strict=True)
        val_loss, val_metrics = _val_epoch(model, val_loader, loss_fn, args.type, args.device)
        epoch = int(ckpt.get("epoch", -1))

        eval_rows.append(
            {
                "checkpoint": str(ckpt_path),
                "epoch": epoch,
                "val_loss": val_loss,
                "metrics": val_metrics,
            }
        )
        if epoch in history_by_epoch:
            history_by_epoch[epoch]["val_loss"] = val_loss
            history_by_epoch[epoch]["metrics"] = val_metrics

        print(
            f"[eval] {ckpt_path.name}: "
            f"val_total={val_loss.get('total', float('nan')):.4f}  "
            f"Z={val_metrics.get('z_error_m', float('nan')):.3f}  "
            f"ADD-S={val_metrics.get('adds_m', float('nan')):.3f}"
        )

    if history_by_epoch:
        merged_history = [history_by_epoch[k] for k in sorted(history_by_epoch)]
        history_path.write_text(json.dumps(merged_history, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[eval] merged history updated → {history_path}")

    out_path = ckpt_dir / "eval_series.json"
    out_path.write_text(json.dumps(eval_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] evaluation summary saved → {out_path}")


if __name__ == "__main__":
    main()

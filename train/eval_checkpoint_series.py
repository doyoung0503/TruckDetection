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
import sys
from pathlib import Path

import torch

from train.dataset import make_dataloaders
from train.models import build_smoke_model
from train.smoke_loss import build_smoke_loss
from train.smoke_trainer import DATASET_ROOT, RESULTS_DIR, _val_epoch

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_SMOKE_DIR = ROOT / "SMOKE-master"
if str(OFFICIAL_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_SMOKE_DIR))

from smoke.utils.model_serialization import load_state_dict as official_align_load_state_dict


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
    if stem.startswith("model_") and stem[6:].isdigit():
        return int(stem[6:]), stem
    if stem == "last":
        return 10**9, stem
    if stem == "best":
        return 10**9 + 1, stem
    if stem == "model_final":
        return 10**9 + 2, stem
    return 10**9 + 3, stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved checkpoint series on the val split.")
    p.add_argument("--type", required=True, choices=["baseline", "geometry", "baseline_depth", "geometry_aux"])
    p.add_argument("--dataset-root", default=DATASET_ROOT, help="Converted KITTI dataset root")
    p.add_argument("--results-dir", default=str(RESULTS_DIR), help="Training results root")
    p.add_argument("--checkpoint-dir", default=None, help="Direct checkpoint directory override")
    p.add_argument("--device", default=_get_device(), help="Evaluation device")
    p.add_argument("--batch", type=int, default=8, help="Validation batch size")
    p.add_argument("--workers", type=int, default=4, help="Validation dataloader workers")
    p.add_argument("--checkpoint-glob", default="*.pth", help="Checkpoint glob inside the checkpoint dir")
    p.add_argument("--include-last", action="store_true", help="Also evaluate last.pt if present")
    p.add_argument("--include-best", action="store_true", help="Also evaluate best.pt if present")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir is not None else (results_dir / args.type)
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
    model_final = ckpt_dir / "model_final.pth"
    if model_final.exists():
        checkpoints.append(model_final)

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

    history_path = ckpt_dir / "history_eval.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = []

    history_by_iteration = {int(h["iteration"]): h for h in history if "iteration" in h}
    meta_path = ckpt_dir / "run_meta.json"
    iters_per_epoch = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        iters_per_epoch = meta.get("iters_per_epoch")
    eval_rows: list[dict] = []

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=args.device)
        if args.type == "baseline":
            official_align_load_state_dict(model, ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"], strict=True)
        val_loss, val_metrics = _val_epoch(model, val_loader, loss_fn, args.type, args.device)
        iteration = int(ckpt.get("iteration", _checkpoint_key(ckpt_path)[0]))
        epoch_equivalent = None
        if iters_per_epoch:
            epoch_equivalent = iteration / max(iters_per_epoch, 1)

        eval_rows.append(
            {
                "checkpoint": str(ckpt_path),
                "iteration": iteration,
                "epoch_equivalent": epoch_equivalent,
                "val_loss": val_loss,
                "metrics": val_metrics,
            }
        )
        if iteration in history_by_iteration:
            history_by_iteration[iteration]["val_loss"] = val_loss
            history_by_iteration[iteration]["metrics"] = val_metrics

        print(
            f"[eval] {ckpt_path.name}: "
            f"iter={iteration}  "
            f"val_total={val_loss.get('total', float('nan')):.4f}  "
            f"Z={val_metrics.get('z_error_m', float('nan')):.3f}  "
            f"ADD-S={val_metrics.get('adds_m', float('nan')):.3f}"
        )

    if history_by_iteration:
        merged_history = [history_by_iteration[k] for k in sorted(history_by_iteration)]
        history_path.write_text(json.dumps(merged_history, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[eval] merged history updated → {history_path}")

    out_path = ckpt_dir / "eval_series.json"
    out_path.write_text(json.dumps(eval_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] evaluation summary saved → {out_path}")


if __name__ == "__main__":
    main()

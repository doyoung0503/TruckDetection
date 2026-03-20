"""
Run the DoF-restricted geometry model with an official SMOKE-style trainer.

Design goal:
  - `baseline` stays on untouched SMOKE-master training code.
  - `geometry` uses the same optimizer / scheduler / checkpointer / iteration
    trainer contract as SMOKE, while only swapping the detector head/loss.
  - The only intentional deviation is the dataset loader: we feed batches from
    this project instead of SMOKE's KITTI loader.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_SMOKE_DIR = ROOT / "SMOKE-master"
if str(OFFICIAL_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_SMOKE_DIR))

from smoke.engine.trainer import do_train
from smoke.solver.build import make_lr_scheduler, make_optimizer
from smoke.utils.check_point import DetectronCheckpointer
from smoke.utils.logger import setup_logger

from train.dataset import make_dataloaders
from train.models import build_smoke_model
from train.run_official_smoke_baseline import _link_dataset, _validate_smoke_repo
from train.smoke_loss import build_official_smoke_cfg, build_smoke_loss


DEFAULT_DATASET_ROOT = ROOT / "datasets" / "v3" / "kitti_smoke_1280x384_lb"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "geometry_smoke"
DEFAULT_MAX_ITER = 25000
DEFAULT_STEPS = (10000, 18000)


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


class _BatchTarget:
    """A tiny target wrapper so official trainer code can call `.to(device)`."""

    def __init__(self, batch: dict[str, Any]):
        self.batch = batch

    def to(self, device: torch.device):
        return _BatchTarget(_move_to_device(self.batch, device))


class _InfiniteOfficialLoader:
    """
    Repeat the project dataloader indefinitely so official iteration-based
    training semantics stay intact across multiple epochs.
    """

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        while True:
            for batch in self.loader:
                yield {
                    "images": batch["image"],
                    "targets": [_BatchTarget(batch)],
                }


class GeometryDetector(nn.Module):
    """
    Official-SMOKE-compatible training wrapper.

    Training: returns a tensor loss dict, just like SMOKE's detector.
    Eval: returns the raw geometry outputs for downstream evaluation scripts.
    """

    def __init__(self):
        super().__init__()
        self.model = build_smoke_model("geometry", pretrained=True)
        self.loss_fn = build_smoke_loss("geometry")

    def forward(self, images, targets=None):
        outputs = self.model(images)
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            if not isinstance(targets, list) or not targets:
                raise TypeError("GeometryDetector expects targets as a non-empty list")
            batch_target = targets[0]
            if not isinstance(batch_target, _BatchTarget):
                raise TypeError("GeometryDetector expects _BatchTarget entries")
            _, tensor_terms, _ = self.loss_fn.compute_loss_terms(outputs, batch_target.batch)
            return tensor_terms
        return outputs

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)


def _build_cfg(
    *,
    device: str,
    batch_size: int,
    max_iter: int,
    steps: tuple[int, int],
    checkpoint_period: int,
    output_dir: Path,
) -> Any:
    cfg = build_official_smoke_cfg(device=device)
    cfg.defrost()
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.MAX_ITERATION = max_iter
    cfg.SOLVER.STEPS = steps
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train geometry with an official-SMOKE-style trainer.")
    p.add_argument("--smoke-dir", type=Path, default=OFFICIAL_SMOKE_DIR, help="Path to SMOKE-master.")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Converted KITTI dataset root.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for checkpoints/logs.")
    p.add_argument("--device", default=_get_device(), help="Training device.")
    p.add_argument("--batch", type=int, default=8, help="Batch size.")
    p.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Official-style max iteration.")
    p.add_argument(
        "--steps",
        type=int,
        nargs=2,
        default=DEFAULT_STEPS,
        metavar=("STEP1", "STEP2"),
        help="Official-style scheduler milestones.",
    )
    p.add_argument("--checkpoint-period", type=int, default=DEFAULT_STEPS[0], help="Checkpoint period in iterations.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    smoke_dir = args.smoke_dir.resolve()
    _validate_smoke_repo(smoke_dir)
    dataset_root = args.dataset_root.resolve()
    _link_dataset(smoke_dir, dataset_root)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(output_dir), 0)
    logger.info("Launching geometry training with file logging enabled.")

    train_loader, _ = make_dataloaders(
        root=str(dataset_root),
        model_type="geometry",
        batch_size=args.batch,
        num_workers=args.workers,
        augment=True,
    )
    iter_loader = _InfiniteOfficialLoader(train_loader)
    iters_per_epoch = len(train_loader)
    max_iter = args.max_iter
    step_1, step_2 = args.steps
    checkpoint_period = args.checkpoint_period

    cfg = _build_cfg(
        device=args.device,
        batch_size=args.batch,
        max_iter=max_iter,
        steps=(step_1, step_2),
        checkpoint_period=checkpoint_period,
        output_dir=output_dir,
    )

    model = GeometryDetector().to(torch.device(args.device))
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    arguments = {"iteration": 0}
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, str(output_dir), save_to_disk=True
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    meta = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "batch": args.batch,
        "workers": args.workers,
        "iters_per_epoch": iters_per_epoch,
        "epochs_equivalent": (max_iter / max(iters_per_epoch, 1)),
        "max_iteration": max_iter,
        "steps": [step_1, step_2],
        "checkpoint_period": checkpoint_period,
        "device": args.device,
        "seed": args.seed,
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[geometry-smoke] cwd:", smoke_dir)
    print("[geometry-smoke] dataset:", dataset_root)
    print("[geometry-smoke] output:", output_dir)
    print("[geometry-smoke] batch:", args.batch)
    print("[geometry-smoke] iters/epoch:", iters_per_epoch)
    print("[geometry-smoke] max_iter:", max_iter)
    print("[geometry-smoke] steps:", (step_1, step_2))
    print("[geometry-smoke] checkpoint_period:", checkpoint_period)
    logging.getLogger("smoke").info(
        "geometry config | dataset=%s output=%s batch=%d max_iter=%d steps=%s checkpoint_period=%d",
        dataset_root,
        output_dir,
        args.batch,
        max_iter,
        (step_1, step_2),
        checkpoint_period,
    )

    do_train(
        cfg=cfg,
        distributed=False,
        model=model,
        data_loader=iter_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpointer=checkpointer,
        device=torch.device(args.device),
        checkpoint_period=checkpoint_period,
        arguments=arguments,
    )


if __name__ == "__main__":
    main()

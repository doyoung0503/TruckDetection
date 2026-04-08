#!/usr/bin/env python
"""Run a one-batch FCOS3D dry-run on the v3 KITTI-mono dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/dy-jang/projects/TruckDetection-main/train/mmdet3d_configs/fcos3d_r101_caffe_dcn_fpn_v3_mono.py",
    )
    parser.add_argument(
        "--mmdet3d-root",
        default="/home/dy-jang/projects/mmdetection3d",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    return parser.parse_args()


def scalarize(loss_dict: dict) -> dict[str, float]:
    out = {}
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
            out[key] = float(sum(v.detach().cpu().item() for v in value))
    return out


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.mmdet3d_root)))

    from mmdet3d.registry import DATASETS, MODELS

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    samples = [dataset[i] for i in range(args.batch_size)]
    batch = pseudo_collate(samples)

    model = MODELS.build(cfg.model)
    model.init_weights()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    data = model.data_preprocessor(batch, training=True)
    loss_dict = model._run_forward(data, mode="loss")
    total_loss = sum(
        value for key, value in loss_dict.items()
        if isinstance(value, torch.Tensor) and "loss" in key)
    total_loss.backward()

    print("device", device)
    print("dataset_len", len(dataset))
    print("losses", scalarize(loss_dict))


if __name__ == "__main__":
    main()

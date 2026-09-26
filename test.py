from __future__ import annotations

import argparse
import math
import statistics

import torch

from src.data.scribble_dataset import ScribbleDataset
from src.diffusion.residual_schedule import ResidualDiffusionSchedule
from src.utils.common import load_cfg, seed_all
from src.utils.metrics import batch_metrics
from src.utils.train_utils import make_loader


def mean_sd(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return float("nan"), float("nan")
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.fmean(values), sd


def build_dataset(client, cfg):
    test = client["test"]
    sparse = cfg["sparse"]

    return ScribbleDataset(
        image_dir=test["images"],
        mask_dir=test["masks"],
        image_size=cfg["data"]["image_size"],
        num_classes=cfg["data"]["num_classes"],
        mask_values=cfg["data"].get("mask_values"),
        sparse_mode=sparse["mode"],
        points_per_class=sparse.get("points_per_class", 8),
        mask_suffix=client.get("mask_suffix", ""),
        augment=False,
        sparse_seed=sparse.get("seed", cfg.get("seed", 42)),
        scribble_width=sparse.get("scribble_width", 0),
        bg_inner_margin=sparse.get("bg_inner_margin", 3),
        bg_outer_margin=sparse.get("bg_outer_margin", 18),
        bg_num_branches=sparse.get("bg_num_branches", 2),
    )


def load_stage1(path, cfg, sparse_channels, device):
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint_cfg = checkpoint.get("cfg", cfg)
    model_cfg = checkpoint_cfg.get("model", cfg.get("model", {}))

    condition_channels = int(checkpoint.get("condition_channels", sparse_channels))

    if condition_channels != sparse_channels:
        raise RuntimeError(
            f"Stage 1 expects {condition_channels} condition channels, "
            f"but test data has {sparse_channels}"
        )

    model = ConditionedStage1UNet(
        base_channels=int(model_cfg.get("base_channels", model_cfg.get("width", 32))),
        feature_channels=int(model_cfg.get("feature_channels", 32)),
        image_channels=3,
        num_classes=int(cfg["data"]["num_classes"]),
        condition_channels=condition_channels,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


@torch.no_grad()
def evaluate(stage1, stage2, loader, schedule, device):
    dice, iou, hd95 = [], [], []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        sparse = batch["sparse"].to(device, non_blocking=True)

        p1 = torch.sigmoid(stage1(image, sparse))
        residual, timestep = schedule.zero_start(p1)
        correction = stage2(residual, timestep, image, p1, sparse)
        prediction = (p1 + correction).clamp(0.0, 1.0)

        d, j, h = batch_metrics(prediction, target)

        dice.append(d)
        iou.append(j)
        hd95.append(h)

    d_mean, d_sd = mean_sd(dice)
    i_mean, i_sd = mean_sd(iou)
    h_mean, h_sd = mean_sd(hd95)

    return {
        "dice": d_mean,
        "dice_sd": d_sd,
        "iou": i_mean,
        "iou_sd": i_sd,
        "hd95": h_mean,
        "hd95_sd": h_sd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breast_tumor.yaml")
    # parser.add_argument("--stage1_checkpoint", default="checkpoints/stage1_checkpoint.pt")
    # parser.add_argument("--stage2_checkpoint", default="checkpoints/stage2_checkpoint.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    seed_all(cfg.get("seed", 42))

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    # checkpoint = torch.load(args.stage2_checkpoint, map_location="cpu")

    datasets = [ build_dataset(client, cfg) for client in cfg["clients"] ]

    loaders = [make_loader(dataset, 1, False, cfg.get("workers", 4)) for dataset in datasets ]

    sparse_channels = int(datasets[0][0]["sparse"].shape[0])
    num_classes = int(cfg["data"]["num_classes"])

    stage1 = torch.jit.load("checkpoints/stage1_model.pth", map_location=device,)
    stage1.eval()
    stage2 = torch.jit.load("checkpoints/stage2_model.pth", map_location=device,)
    stage2.eval()

    schedule = ResidualDiffusionSchedule( steps=int(cfg["stage2"].get("diffusion_steps", 20)),
        device=device, min_alpha_bar=float( cfg["stage2"].get("min_alpha_bar", 1e-4) ), )

    results = [ evaluate(stage1, stage2, loader, schedule, device)
        for loader in loaders ]

    for i, result in enumerate(results, 1):
        print(
            f"C{i}: "
            f"Dice {result['dice'] * 100:.2f} ± {result['dice_sd'] * 100:.2f} | "
            f"IoU {result['iou'] * 100:.2f} ± {result['iou_sd'] * 100:.2f} | "
            f"HD95 {result['hd95']:.2f} ± {result['hd95_sd']:.2f}"
        )

    dice_mean, dice_sd = mean_sd([r["dice"] for r in results])
    iou_mean, iou_sd = mean_sd([r["iou"] for r in results])
    hd95_mean, hd95_sd = mean_sd([r["hd95"] for r in results])

    print(
        f"Avg: "
        f"Dice {dice_mean * 100:.2f} ± {dice_sd * 100:.2f} | "
        f"IoU {iou_mean * 100:.2f} ± {iou_sd * 100:.2f} | "
        f"HD95 {hd95_mean:.2f} ± {hd95_sd:.2f}"
    )


if __name__ == "__main__":
    main()
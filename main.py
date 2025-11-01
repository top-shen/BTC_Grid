# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import GridKlineDataset
from model import PatchTST

from utils.io import resolve_csv_path, ensure_dir
from utils.data import (
    compute_and_plot_delta_distribution,
    build_bucketed_sampler,
    baseline_last_mse,
)
from utils.optim import build_adamw_with_groups
from utils.trainer_quantile import QuantileTrainer
from utils.plot import (
    scatter_preds_vs_actuals,
    confusion_matrix_plot,
    plot_last_curves,
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

parser = argparse.ArgumentParser()
# CSV will be resolved via utils.io
parser.add_argument("--csv", type=str, default="./BTCUSDT_5m.csv")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--bins", type=int, default=60)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=1e-3)

args = parser.parse_args()

# Ensure results directory exists for all outputs (figures, ckpt, csv, etc.)
RESULTS_DIR = "results"
ensure_dir(RESULTS_DIR)

NUM_BINS = int(args.bins)

# Load CSV with fallback to data/
csv_path = resolve_csv_path(args.csv)
print(f"Loading CSV: {csv_path}")
df = pd.read_csv(csv_path)

# Split (time series: no shuffle)
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# Datasets/loaders
train_dataset = GridKlineDataset(train_df, num_bins=NUM_BINS)
test_dataset = GridKlineDataset(test_df, num_bins=NUM_BINS)
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

# 固定使用 PatchTST（根据数据通道数进行初始化）
BEST_CKPT_PATH = os.path.join(RESULTS_DIR, "best.pt")
model = PatchTST(vocab_size=NUM_BINS, num_channels=train_dataset.num_channels).to(device)
optimizer = build_adamw_with_groups(model, lr=args.lr, weight_decay=args.weight_decay)

# Diagnostics
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model:", model.__class__.__name__)
print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")
pg0 = optimizer.param_groups[0] if optimizer.param_groups else {}
print(
    "Optimizer config:",
    {"name": optimizer.__class__.__name__, "lr": pg0.get("lr"), "weight_decay": pg0.get("weight_decay")},
)

# Delta distribution diagnostics and scaling
train_stats = compute_and_plot_delta_distribution(
    train_loader, "train", os.path.join(RESULTS_DIR, "delta_distribution_train.png")
)
compute_and_plot_delta_distribution(
    test_loader, "test", os.path.join(RESULTS_DIR, "delta_distribution_test.png")
)

# Old Student-t hyperparams are not used in quantile training; keep delta stats plot for reference only.

# Weighted sampler balancing pos/neg by sign of (y - last)
print("Building sign-balanced sampling weights (delta sign)...")
sampler = build_bucketed_sampler(train_dataset)
if sampler is not None:
    train_loader = DataLoader(train_dataset, batch_size=args.batch, sampler=sampler)
    print("Train loader switched to WeightedRandomSampler.")
else:
    print("No training samples to build bucketed sampler; keep original shuffle loader.")

# Training
best_val = float("inf")
no_improve_count = 0
patience = 1
num_epochs = int(args.epochs)

trainer = QuantileTrainer(
    model=model,
    device=device,
    quantiles=(0.1, 0.5, 0.9),
)

for epoch in range(num_epochs):
    train_ql = trainer.train_epoch(train_loader, optimizer)
    print(f"Epoch {epoch + 1}: train_qloss = {train_ql:.4f}")

    val = trainer.validate(test_loader, num_bins=NUM_BINS)
    cov_info = ", ".join([f"cov@{k.split('@')[-1]}={v:.3f}" for k, v in val.items() if k.startswith('cov@')])
    print(
        f"Epoch {epoch + 1}: val_qloss={val['qloss']:.4f}, MAE={val['mae']:.4f}, MSE={val['mse']:.4f}" +
        (f", {cov_info}" if cov_info else "")
    )

    if val["qloss"] < best_val:
        best_val = val["qloss"]
        no_improve_count = 0
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_qloss": float(val["qloss"]),
                "num_bins": NUM_BINS,
            },
            BEST_CKPT_PATH,
        )
        print(f"Saved new best model to {BEST_CKPT_PATH} (val_qloss={val['qloss']:.4f})")
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(
                f"Early stop at epoch {epoch + 1}: val qloss did not improve for {patience} epochs"
            )
            break

# Evaluation & plots
all_preds, all_reals = trainer.collect_preds(test_loader, NUM_BINS) # type: ignore
scatter_preds_vs_actuals(
    all_preds, all_reals, os.path.join(RESULTS_DIR, "scatter_preds_vs_actuals.png")
)
confusion_matrix_plot(
    all_preds, all_reals, NUM_BINS, os.path.join(RESULTS_DIR, "confusion_matrix.png")
)

mse = float(np.mean((all_preds - all_reals) ** 2))
print("\nVisualizations saved under results/: scatter_preds_vs_actuals.png, confusion_matrix.png")
print(f"Test MSE: {mse:.4f}")
print("First 10 Actual Grids:", all_reals[:10])
print("First 10 Predicted Grids:", all_preds[:10])

# Last N k-lines plot
last_N = 2000
last_df = df.iloc[-last_N:].copy()
last_dataset = GridKlineDataset(last_df, num_bins=NUM_BINS)
last_loader = DataLoader(last_dataset, batch_size=256, shuffle=False)
last_preds, last_reals, last_grids = trainer.collect_preds( # type: ignore
    last_loader, NUM_BINS, return_last=True
)
plot_last_curves(
    reals=last_reals,
    preds=last_preds,
    lasts=last_grids,
    out_path=os.path.join(RESULTS_DIR, "last_2000_pred_vs_actual.png"),
    title=f"Last {last_N} K-lines: Predicted vs Actual + Last Grids",
)

# Baseline
baseline_mse = baseline_last_mse(test_loader)
print(f"Baseline (last-grid) on test: MSE={baseline_mse:.4f}")

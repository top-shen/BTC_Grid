# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import GridKlineDataset
from model_factory import build_model, build_model_config_from_args
from utils.io import resolve_csv_path, ensure_dir
from utils.data import compute_and_plot_delta_distribution, build_bucketed_sampler, baseline_last_mse
from utils.optim import build_adamw_with_groups
from utils.trainer_quantile import QuantileTrainer
from utils.plot import scatter_preds_vs_actuals, confusion_matrix_plot, plot_last_curves


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="./BTCUSDT_5m.csv")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--bins", type=int, default=60)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=1e-3)

parser.add_argument("--input_mode", type=str, default="baseline", choices=["baseline", "tokenizer", "vq"])
parser.add_argument("--use_vq", type=str2bool, default=True)
parser.add_argument("--lambda_vq", type=float, default=0.1)
parser.add_argument("--fusion_mode", type=str, default="ze_zq_fusion", choices=["ze_only", "zq_only", "ze_zq_fusion"])
parser.add_argument("--zq_pool_mode", type=str, default="mean", choices=["mean", "attn_pool", "full"])
parser.add_argument("--patchtst_token_mode", type=str, default="repatch", choices=["repatch", "direct_tokens"])
parser.add_argument("--tokenizer_input_mode", type=str, default="grid_cont", choices=["grid_cont", "raw_ohlcv"])

parser.add_argument("--vq_local_window", type=int, default=24)
parser.add_argument("--vq_local_stride", type=int, default=12)
parser.add_argument("--vq_patch_len", type=int, default=4)
parser.add_argument("--vq_patch_stride", type=int, default=4)
parser.add_argument("--vq_codebook_size", type=int, default=256)
parser.add_argument("--vq_encoder_layers", type=int, default=2)
parser.add_argument("--vq_decoder_layers", type=int, default=1)
parser.add_argument("--vq_heads", type=int, default=4)
parser.add_argument("--fusion_num_layers", type=int, default=1)
parser.add_argument("--fusion_num_heads", type=int, default=4)
parser.add_argument("--vq_mlp_ratio", type=float, default=4.0)
parser.add_argument("--vq_dropout", type=float, default=0.1)
parser.add_argument("--vq_recon_weight", type=float, default=1.0)
parser.add_argument("--vq_commit_weight", type=float, default=0.25)
parser.add_argument("--vq_orthogonal_weight", type=float, default=0.0)
parser.add_argument("--vq_diversity_weight", type=float, default=0.0)
args = parser.parse_args()

RESULTS_DIR = "results"
ensure_dir(RESULTS_DIR)
NUM_BINS = int(args.bins)
normalized_input_mode = "tokenizer" if args.input_mode == "vq" else args.input_mode
use_tokenizer = normalized_input_mode != "baseline"

if not use_tokenizer:
    args.use_vq = False
if use_tokenizer and (not args.use_vq) and args.fusion_mode != "ze_only":
    raise ValueError("When --use_vq false, please use --fusion_mode ze_only for the continuous-only ablation.")

csv_path = resolve_csv_path(args.csv)
print(f"Loading CSV: {csv_path}")
df = pd.read_csv(csv_path)
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

train_dataset = GridKlineDataset(
    train_df,
    num_bins=NUM_BINS,
    return_continuous=use_tokenizer,
    tokenizer_input_mode=args.tokenizer_input_mode,
)
test_dataset = GridKlineDataset(
    test_df,
    num_bins=NUM_BINS,
    return_continuous=use_tokenizer,
    tokenizer_input_mode=args.tokenizer_input_mode,
)
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

BEST_CKPT_PATH = os.path.join(RESULTS_DIR, "best.pt")
model_config = build_model_config_from_args(args)
model = build_model(
    num_bins=NUM_BINS,
    num_channels=train_dataset.num_channels,
    feature_dim=train_dataset.feature_dim,
    model_config=model_config,
).to(device)
optimizer = build_adamw_with_groups(model, lr=args.lr, weight_decay=args.weight_decay)

print("Model:", model.__class__.__name__)
print("Input mode:", model_config["input_mode"])
print("use_vq:", model_config["use_vq"])
print("fusion_mode:", model_config["fusion_mode"])
print("zq_pool_mode:", model_config["zq_pool_mode"])
print("patchtst_token_mode:", model_config["patchtst_token_mode"])
print("tokenizer_input_mode:", model_config["tokenizer_input_mode"])
print("Tokenizer feature dim:", train_dataset.feature_dim)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")
pg0 = optimizer.param_groups[0] if optimizer.param_groups else {}
print("Optimizer config:", {"name": optimizer.__class__.__name__, "lr": pg0.get("lr"), "weight_decay": pg0.get("weight_decay")})

compute_and_plot_delta_distribution(train_loader, "train", os.path.join(RESULTS_DIR, "delta_distribution_train.png"))
compute_and_plot_delta_distribution(test_loader, "test", os.path.join(RESULTS_DIR, "delta_distribution_test.png"))

print("Building sign-balanced sampling weights (delta sign)...")
sampler = build_bucketed_sampler(train_dataset)
if sampler is not None:
    train_loader = DataLoader(train_dataset, batch_size=args.batch, sampler=sampler)
    print("Train loader switched to WeightedRandomSampler.")
else:
    print("No training samples to build bucketed sampler; keep original shuffle loader.")

best_val = float("inf")
no_improve_count = 0
patience = int(args.patience)
trainer = QuantileTrainer(model=model, device=device, quantiles=(0.1, 0.5, 0.9), lambda_vq=args.lambda_vq)

for epoch in range(int(args.epochs)):
    train_stats = trainer.train_epoch(train_loader, optimizer)
    msg = (
        f"Epoch {epoch + 1}: train_pred_qloss={train_stats['pred_qloss']:.4f}, "
        f"train_total={train_stats['total_loss']:.4f}"
    )
    if use_tokenizer:
        msg += (
            f", train_vq={train_stats['vq_loss']:.4f}, recon={train_stats['recon_loss']:.4f}, "
            f"codebook={train_stats['codebook_loss']:.4f}, commit={train_stats['commitment_loss']:.4f}, "
            f"used_codes={train_stats['used_codes']:.2f}, top1_freq={train_stats['top1_code_freq']:.3f}"
        )
        if args.vq_diversity_weight > 0:
            msg += f", diversity={train_stats['diversity_loss']:.4f}"
        if args.vq_orthogonal_weight > 0:
            msg += f", orth={train_stats['orthogonal_loss']:.4f}"
    print(msg)

    val = trainer.validate(test_loader, num_bins=NUM_BINS)
    cov_info = ", ".join([f"cov@{k.split('@')[-1]}={v:.3f}" for k, v in val.items() if k.startswith('cov@')])
    msg = (
        f"Epoch {epoch + 1}: val_qloss={val['qloss']:.4f}, val_total={val['total_loss']:.4f}, "
        f"MAE={val['mae']:.4f}, MSE={val['mse']:.4f}"
    )
    if use_tokenizer:
        msg += (
            f", val_vq={val['vq_loss']:.4f}, val_recon={val['recon_loss']:.4f}, "
            f"used_codes={val['used_codes']:.2f}, top1_freq={val['top1_code_freq']:.3f}"
        )
    if cov_info:
        msg += f", {cov_info}"
    print(msg)

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
                "model_config": model_config,
                "feature_dim": train_dataset.feature_dim,
            },
            BEST_CKPT_PATH,
        )
        print(f"Saved new best model to {BEST_CKPT_PATH} (val_qloss={val['qloss']:.4f})")
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stop at epoch {epoch + 1}: val qloss did not improve for {patience} epochs")
            break

all_preds, all_reals = trainer.collect_preds(test_loader, NUM_BINS)
scatter_preds_vs_actuals(all_preds, all_reals, os.path.join(RESULTS_DIR, "scatter_preds_vs_actuals.png"))
confusion_matrix_plot(all_preds, all_reals, NUM_BINS, os.path.join(RESULTS_DIR, "confusion_matrix.png"))

mse = float(np.mean((all_preds - all_reals) ** 2))
print("\nVisualizations saved under results/: scatter_preds_vs_actuals.png, confusion_matrix.png")
print(f"Test MSE: {mse:.4f}")
print("First 10 Actual Grids:", all_reals[:10])
print("First 10 Predicted Grids:", all_preds[:10])

last_N = 2000
last_df = df.iloc[-last_N:].copy()
last_dataset = GridKlineDataset(
    last_df,
    num_bins=NUM_BINS,
    return_continuous=use_tokenizer,
    tokenizer_input_mode=args.tokenizer_input_mode,
)
last_loader = DataLoader(last_dataset, batch_size=256, shuffle=False)
last_preds, last_reals, last_grids = trainer.collect_preds(last_loader, NUM_BINS, return_last=True)
plot_last_curves(
    reals=last_reals,
    preds=last_preds,
    lasts=last_grids,
    out_path=os.path.join(RESULTS_DIR, "last_2000_pred_vs_actual.png"),
    title=f"Last {last_N} K-lines: Predicted vs Actual + Last Grids",
)

baseline_mse = baseline_last_mse(test_loader)
print(f"Baseline (last-grid) on test: MSE={baseline_mse:.4f}")

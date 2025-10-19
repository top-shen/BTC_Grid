# -*- coding: utf-8 -*-
"""
Standalone test script that loads the best checkpoint and simulates trading.

Rules:
- Compute model-predicted delta mu (relative to last grid).
- Enter a trade when |mu| is in the top quantile (default 80%).
  * If mu > 0: long; profit = (y_abs - last).
  * If mu < 0: short; profit = -(y_abs - last).
- Profit is measured in grid units as requested.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import GridKlineDataset
from model import GridTransformer, PatchTST
import torch.nn.functional as F


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def last_main_grid(batch_x: torch.Tensor) -> torch.Tensor:
    # Expect x: [B,T,4]; take last step of main channel
    assert batch_x.dim() == 3 and batch_x.size(-1) == 4, "expect x shape [B,T,4]"
    return batch_x[:, -1, 0].float()


def build_model(kind: str, vocab_size: int) -> torch.nn.Module:
    if kind == "grid":
        return GridTransformer(vocab_size=vocab_size).to(device)
    else:
        return PatchTST(vocab_size=vocab_size).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="./BTCUSDT_5m.csv")
    p.add_argument("--ckpt", type=str, default="best_patchtst.pt")
    p.add_argument("--quantile", type=float, default=0.80, help="abs(mu) quantile threshold for entry")
    p.add_argument("--window", type=int, default=5000, help="rolling window length for quantile; 0 means use all past history")
    p.add_argument("--min_hist", type=int, default=1000, help="minimum number of past predictions required before taking trades")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--out_csv", type=str, default="trades.csv")
    p.add_argument("--no_plot", action="store_true", help="do not save equity curve plot")
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device)
    model_type = ckpt.get("model_type", "patchtst")
    num_bins = int(ckpt.get("num_bins", 60))
    model = build_model(model_type, vocab_size=num_bins)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded checkpoint: {args.ckpt} (epoch={ckpt.get('epoch','?')}, val_nll={ckpt.get('val_nll','?')})")
    print(f"Model type: {model_type}, num_bins={num_bins}")

    # Prepare data (keep the same split logic as training: last 20% as test)
    df = pd.read_csv(args.csv)
    _, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    test_dataset = GridKlineDataset(test_df, num_bins=num_bins)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    preds_mu, last_vals, y_abs_vals = [], [], []
    with torch.no_grad():
        for x, y_float, _ in test_loader:
            x = x.to(device)
            y_abs = y_float.to(device)
            last = last_main_grid(x).to(device)
            out = model(x)
            mu = out[:, 0]  # predicted delta
            preds_mu.append(mu.cpu().numpy())
            last_vals.append(last.cpu().numpy())
            y_abs_vals.append(y_abs.cpu().numpy())

    mu = np.concatenate(preds_mu).astype(np.float32)
    last = np.concatenate(last_vals).astype(np.float32)
    y_abs = np.concatenate(y_abs_vals).astype(np.float32)
    delta_true = (y_abs - last).astype(np.float32)

    # Rolling, causal quantile thresholding (no look-ahead):
    # At time t, compute q from abs(mu) of indices [max(0, t-window), t) only.
    abs_mu = np.abs(mu)
    N = len(mu)
    W = int(args.window)
    min_hist = int(min(max(1, args.min_hist), max(1, N - 1)))  # ensure feasible
    enter = np.zeros(N, dtype=bool)
    q_used = np.full(N, np.nan, dtype=np.float32)
    side = np.sign(mu).astype(np.float32)  # +1 long, -1 short, 0 no-trade
    for t in range(N):
        hist_end = t  # exclusive
        hist_start = max(0, t - W) if W > 0 else 0
        if hist_end - hist_start >= min_hist:
            q = float(np.quantile(abs_mu[hist_start:hist_end], args.quantile))
            q_used[t] = q
            enter[t] = abs_mu[t] >= q
        else:
            enter[t] = False  # not enough history -> no entry
    pos = side * enter.astype(np.float32)

    profit = pos * delta_true  # per-trade profit in grid units
    took = enter.sum()
    long_n = int(((pos > 0).astype(np.int32)).sum())
    short_n = int(((pos < 0).astype(np.int32)).sum())
    win = (profit > 0).sum()
    loss = (profit < 0).sum()
    flat = (profit == 0).sum()
    total_pnl = float(profit.sum())
    avg_pnl = float(profit[enter].mean()) if took > 0 else 0.0
    std_pnl = float(profit[enter].std()) if took > 1 else 0.0
    win_rate = float(win) / float(took) if took > 0 else 0.0

    print(f"Rolling quantile: q={args.quantile}, window={W if W>0 else 'ALL-PAST'}, min_hist={min_hist}")
    used = q_used[~np.isnan(q_used)]
    if used.size > 0:
        print(f"Threshold stats (abs mu): count={used.size}, min={used.min():.4f}, med={np.median(used):.4f}, max={used.max():.4f}")
    print(f"Trades: {int(took)} (long={long_n}, short={short_n}) out of {len(mu)} samples")
    print(f"Total PnL (grids): {total_pnl:.3f}; Avg/trade: {avg_pnl:.3f}; Win rate: {win_rate:.3f}")
    if std_pnl > 1e-12:
        sharpe_like = avg_pnl / std_pnl
        print(f"Sharpe-like (mean/std per trade): {sharpe_like:.3f}")

    # Save trades to csv for inspection
    idx = np.arange(len(mu))
    rows = pd.DataFrame(
        {
            "idx": idx,
            "mu": mu,
            "abs_mu": np.abs(mu),
            "last": last,
            "y_abs": y_abs,
            "delta_true": delta_true,
            "enter": enter.astype(np.int32),
            "side": np.where(pos > 0, 1, np.where(pos < 0, -1, 0)),
            "profit": profit,
            "q_hist": q_used,
        }
    )
    rows.to_csv(args.out_csv, index=False)
    print(f"Saved trades to {args.out_csv}")

    # Optional equity curve
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            equity = np.cumsum(profit)
            plt.figure(figsize=(10, 4))
            plt.plot(equity, label="Equity (grid units)")
            plt.title("Equity Curve of Top-Quantile Strategy")
            plt.xlabel("Sample index (test set order)")
            plt.ylabel("Cumulative profit (grids)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("equity_curve.png")
            plt.close()
            print("Saved equity curve: equity_curve.png")
        except Exception as e:
            print(f"Warning: failed to save equity curve plot: {e}")


if __name__ == "__main__":
    main()


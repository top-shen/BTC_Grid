# -*- coding: utf-8 -*-
"""
Standalone test script that loads the best checkpoint and simulates trading.

Updated rules (quantile head):
- Model outputs 3 quantiles of delta: [q10, q50, q90] by default.
- Enter long when q10 > 0 (delta likely positive); enter short when q90 < 0 (delta likely negative).
- Profit is measured in grid units on the true delta (y_abs - last).
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from dataset import GridKlineDataset
from model import PatchTST
from utils.io import resolve_csv_path
from utils.data import last_main_grid


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Centralize results directory for outputs
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="./BTCUSDT_5m.csv")
    p.add_argument("--ckpt", type=str, default=os.path.join(RESULTS_DIR, "best.pt"))
    # No rolling thresholding; entry is determined by quantile crossings
    p.add_argument("--batch", type=int, default=512)
    # Rolling-quantile rule on q50: go long if q50 > q80(history), short if q50 < q20(history)
    p.add_argument("--roll_win", type=int, default=2000, help="rolling window length for q50 quantiles")
    p.add_argument("--q_low", type=float, default=0.1, help="lower quantile for short entry (e.g., 0.20)")
    p.add_argument("--q_high", type=float, default=0.99, help="upper quantile for long entry (e.g., 0.80)")
    # Save trades CSV under results/ by default
    p.add_argument("--out_csv", type=str, default=os.path.join(RESULTS_DIR, "trades.csv"))
    args = p.parse_args()

    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        alt = os.path.join(RESULTS_DIR, os.path.basename(ckpt_path))
        if os.path.exists(alt):
            print(f"Checkpoint not found at '{ckpt_path}', using '{alt}'")
            ckpt_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: '{args.ckpt}' or '{alt}'")

    ckpt = torch.load(ckpt_path, map_location=device)
    num_bins = int(ckpt.get("num_bins", 60))

    # Prepare data (keep the same split logic as training: last 20% as test)
    csv_path = resolve_csv_path(args.csv)
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    _, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    test_dataset = GridKlineDataset(test_df, num_bins=num_bins)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    model = PatchTST(vocab_size=num_bins, num_channels=test_dataset.num_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path} (epoch={ckpt.get('epoch','?')}, val_nll={ckpt.get('val_nll','?')})")
    print(f"Model: PatchTST, num_bins={num_bins}, channels={test_dataset.num_channels}")

    q10_list, q50_list, q90_list = [], [], []
    last_vals, y_abs_vals = [], []
    with torch.no_grad():
        for x, y_float, _ in test_loader:
            x = x.to(device)
            y_abs = y_float.to(device)
            last = last_main_grid(x).to(device)
            out = model(x)
            # Expect default out_dim=3: [q10, q50, q90] on delta
            if out.size(1) < 3:
                raise RuntimeError(
                    f"Expected model to output 3 quantiles [q10,q50,q90], got shape {tuple(out.shape)}"
                )
            q10_list.append(out[:, 0].cpu().numpy())
            q50_list.append(out[:, 1].cpu().numpy())
            q90_list.append(out[:, 2].cpu().numpy())
            last_vals.append(last.cpu().numpy())
            y_abs_vals.append(y_abs.cpu().numpy())

    q10 = np.concatenate(q10_list).astype(np.float32)
    q50 = np.concatenate(q50_list).astype(np.float32)
    q90 = np.concatenate(q90_list).astype(np.float32)
    last = np.concatenate(last_vals).astype(np.float32)
    y_abs = np.concatenate(y_abs_vals).astype(np.float32)
    delta_true = (y_abs - last).astype(np.float32)

    # Entry rule: compare current q50 to rolling quantiles of past q50 values
    N = len(q50)
    import pandas as _pd
    s_q50 = _pd.Series(q50)
    roll_q_high = s_q50.rolling(window=args.roll_win, min_periods=args.roll_win).quantile(args.q_high).values
    roll_q_low = s_q50.rolling(window=args.roll_win, min_periods=args.roll_win).quantile(args.q_low).values
    enter_long = q50 > roll_q_high  # long if current median delta above historical q80
    enter_short = q50 < roll_q_low  # short if current median delta below historical q20
    side = np.zeros(N, dtype=np.float32)
    side[enter_long] = 1.0
    side[enter_short] = -1.0

    profit = side * delta_true  # per-trade profit in grid units
    took = int((enter_long | enter_short).sum())
    long_n = int(enter_long.sum())
    short_n = int(enter_short.sum())
    total_pnl = float(profit.sum())

    # Per-side stats
    long_profit = profit[enter_long]
    short_profit = profit[enter_short]
    long_pnl = float(long_profit.sum()) if long_n > 0 else 0.0
    short_pnl = float(short_profit.sum()) if short_n > 0 else 0.0
    win = (profit[(enter_long | enter_short)] > 0).sum()
    loss = (profit[(enter_long | enter_short)] < 0).sum()
    flat = (profit[(enter_long | enter_short)] == 0).sum()
    avg_pnl = float(profit[(enter_long | enter_short)].mean()) if took > 0 else 0.0
    std_pnl = float(profit[(enter_long | enter_short)].std()) if took > 1 else 0.0
    win_rate = float(win) / float(took) if took > 0 else 0.0
    long_win_rate = float((long_profit > 0).sum()) / float(long_n) if long_n > 0 else 0.0
    short_win_rate = float((short_profit > 0).sum()) / float(short_n) if short_n > 0 else 0.0

    print(
        f"Entry: long if q50 > rolling q{int(args.q_high*100)} over {args.roll_win}; "
        f"short if q50 < rolling q{int(args.q_low*100)}"
    )
    print(f"Trades: {took} (long={long_n}, short={short_n}) out of {N} samples")
    print(f"Total PnL (grids): {total_pnl:.3f}; Avg/trade: {avg_pnl:.3f}; Win rate: {win_rate:.3f}")
    print(f"  Long PnL:  {long_pnl:.3f} over {long_n} trades (win rate {long_win_rate:.3f})")
    print(f"  Short PnL: {short_pnl:.3f} over {short_n} trades (win rate {short_win_rate:.3f})")
    if std_pnl > 1e-12:
        sharpe_like = avg_pnl / std_pnl
        print(f"Sharpe-like (mean/std per trade): {sharpe_like:.3f}")

    # Save trades to csv for inspection
    idx = np.arange(N)
    rows = pd.DataFrame(
        {
            "idx": idx,
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "last": last,
            "y_abs": y_abs,
            "delta_true": delta_true,
            "enter_long": enter_long.astype(np.int32),
            "enter_short": enter_short.astype(np.int32),
            "roll_q_high": roll_q_high,
            "roll_q_low": roll_q_low,
            "side": side.astype(np.int32),
            "profit": profit,
        }
    )
    # Ensure parent directory exists for out_csv
    out_dir = os.path.dirname(os.path.abspath(args.out_csv)) or "."
    os.makedirs(out_dir, exist_ok=True)
    rows.to_csv(args.out_csv, index=False)
    print(f"Saved trades to {args.out_csv}")

    # Save equity curve
    equity = np.cumsum(profit)
    plt.figure(figsize=(10, 4))
    plt.plot(equity, label="Equity (grid units)")
    plt.title("Equity Curve (q50 vs rolling q80/q20)")
    plt.xlabel("Sample index (test set order)")
    plt.ylabel("Cumulative profit (grids)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "equity_curve.png"))
    plt.close()
    print(f"Saved equity curve: {os.path.join(RESULTS_DIR, 'equity_curve.png')}")

if __name__ == "__main__":
    main()

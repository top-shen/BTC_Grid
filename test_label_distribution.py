# -*- coding: utf-8 -*-
"""
Quick check for label distribution and clipping rates under current dataset logic.

It replicates GridKlineDataset's target construction to measure:
- y==0 / y==K-1 fractions (edge classes)
- raw index clipping rates: idx_raw < 0 and idx_raw > K-1

Run:
  python tests/test_label_distribution.py \
      --csv ./BTCUSDT_5m.csv --num-bins 60 --window-size 300 --pred-len 50
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_split(df, name: str, num_bins: int, window_size: int, pred_len: int):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    K = int(num_bins)
    edge_lower = 0
    edge_upper = 0
    ys = []

    # Mirror dataset.py logic exactly now:
    # - Range from windowed HIGH (max) and LOW (min)
    # - Split window [low, high] into 20 equal bins (L20..L40 => 20 cells)
    # - Extend 20 bins down (L0..L20) and 20 up (L40..L60) with same delta
    # - Target uses CLOSE price at (t+pred_len-1), quantized to 60 bins [0..59]
    for i in range(len(close) - window_size - pred_len):
        high_w = float(np.max(high[i : i + window_size]))
        low_w = float(np.min(low[i : i + window_size]))
        rng = high_w - low_w
        if rng <= 1e-12:
            rng = 1e-12
        delta = rng / 20.0
        l0 = low_w - 20.0 * delta

        close_future = close[i + window_size + pred_len - 1]
        idx_raw = (close_future - l0) / delta
        idx = int(np.floor(idx_raw))

        if idx < 0:
            edge_lower += 1
        if idx > K - 1:
            edge_upper += 1

        # Clip to [0, K-1] to match dataset targets
        ys.append(min(K - 1, max(0, idx)))

    ys = np.array(ys, dtype=np.int64)
    total = len(ys)
    counts = np.bincount(ys, minlength=K)

    print(f"[{name}] samples={total}")
    print(
        "y==0: %d (%.3f), y==%d: %d (%.3f)"
        % (counts[0], counts[0] / total if total else 0.0, K - 1, counts[-1], counts[-1] / total if total else 0.0)
    )
    print(
        "clipped lower: %d (%.3f), upper: %d (%.3f)"
        % (
            edge_lower,
            edge_lower / total if total else 0.0,
            edge_upper,
            edge_upper / total if total else 0.0,
        )
    )
    # Small head/tail of histogram for a quick glance
    print("label histogram head:", counts[:10].tolist())
    print("label histogram tail:", counts[-10:].tolist())
    print()
    return counts

def plot_label_hist(counts_train, counts_test, K, out_path):
    xs = np.arange(K)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes[0].bar(xs, counts_train, width=1.0, color="steelblue", edgecolor="none")
    axes[0].set_title(f"Train label distribution (K={K})")
    axes[0].set_xlabel("bin")
    axes[0].set_ylabel("count")
    axes[0].set_xlim(-0.5, K - 0.5)

    axes[1].bar(xs, counts_test, width=1.0, color="indianred", edgecolor="none")
    axes[1].set_title("Test label distribution")
    axes[1].set_xlabel("bin")
    axes[1].set_xlim(-0.5, K - 0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved label distribution figure: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./BTCUSDT_5m.csv")
    ap.add_argument("--num-bins", type=int, default=60)
    ap.add_argument("--window-size", type=int, default=300)
    ap.add_argument("--pred-len", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Time-ordered split 80/20 like main.py (no shuffle)
    n = len(df)
    split = int(n * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    counts_train = analyze_split(train_df, "train", args.num_bins, args.window_size, args.pred_len)
    counts_test = analyze_split(test_df, "test", args.num_bins, args.window_size, args.pred_len)

    out_path = "label_distribution.png"
    plot_label_hist(counts_train, counts_test, args.num_bins, out_path)


if __name__ == "__main__":
    main()

from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


BatchTuple = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]


def unpack_batch(batch) -> BatchTuple:
    """Support both baseline batches and tokenizer batches.

    Baseline:
      (x_grid, y_float, y_cls)

    Tokenizer modes:
      (x_grid, x_tok, y_float, y_cls)
    """
    if len(batch) == 3:
        x_grid, y_float, y_cls = batch
        x_tok = None
    elif len(batch) == 4:
        x_grid, x_tok, y_float, y_cls = batch
    else:
        raise ValueError(f"Unexpected batch format with {len(batch)} items")
    return x_grid, x_tok, y_float, y_cls


def last_main_grid(batch_x: torch.Tensor) -> torch.Tensor:
    assert batch_x.dim() == 3 and batch_x.size(-1) >= 1, "expect x shape [B,T,C] with C>=1"
    return batch_x[:, -1, 0].float()


def compute_and_plot_delta_distribution(loader: DataLoader, name: str, out_path: str) -> Dict[str, float]:
    import matplotlib.pyplot as plt

    deltas = []
    with torch.no_grad():
        for batch in loader:
            x_grid, _, y_float, _ = unpack_batch(batch)
            last = last_main_grid(x_grid)
            deltas.append((y_float.float() - last).numpy())
    if not deltas:
        print(f"[{name}] no data for delta distribution")
        return {"count": 0}
    deltas = np.concatenate(deltas, axis=0)

    q = np.percentile(deltas, [5, 25, 50, 75, 95])
    frac_1 = np.mean((deltas >= -1.0) & (deltas <= 1.0))
    frac_2 = np.mean((deltas >= -2.0) & (deltas <= 2.0))
    frac_5 = np.mean((deltas >= -5.0) & (deltas <= 5.0))
    print(
        f"[{name}] delta stats: count={deltas.size}, min={deltas.min():.2f}, max={deltas.max():.2f}, "
        f"mean={deltas.mean():.3f}, median={q[2]:.3f}, p5/25/75/95={q[0]:.2f}/{q[1]:.2f}/{q[3]:.2f}/{q[4]:.2f}, "
        f"|d|<=1:{frac_1:.3f}, |d|<=2:{frac_2:.3f}, |d|<=5:{frac_5:.3f}"
    )

    bins = np.arange(-40.5, 40.5 + 1.0, 1.0).tolist()
    plt.figure(figsize=(8, 4))
    plt.hist(deltas, bins=bins, color="steelblue", edgecolor="none", alpha=0.85)
    plt.title(f"Delta distribution ({name})")
    plt.xlabel("delta (y_abs - last)")
    plt.ylabel("count")
    plt.xlim(-41, 41)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved delta distribution figure: {out_path}")

    abs_d = np.abs(deltas).astype("float64")
    q90_abs = float(np.quantile(abs_d, 0.90))
    std = float(deltas.astype("float64").std())
    return {"count": int(deltas.size), "q90_abs": q90_abs, "std": std}


def build_bucketed_sampler(dataset) -> WeightedRandomSampler:
    tmp_loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    signs = []
    with torch.no_grad():
        for batch in tmp_loader:
            x_grid, _, y_float, _ = unpack_batch(batch)
            last = last_main_grid(x_grid)
            d = (y_float.float() - last).numpy()
            signs.append(np.sign(d).astype(np.int8))

    signs = np.concatenate(signs)
    n = int(signs.size)
    n_pos = int((signs > 0).sum())
    n_neg = int((signs < 0).sum())
    n_zero = n - n_pos - n_neg

    if n > 0:
        print(
            "Sign split (train): "
            f"total={n}, pos={n_pos} ({n_pos/max(n,1):.3f}), neg={n_neg} ({n_neg/max(n,1):.3f}), zero={n_zero} ({n_zero/max(n,1):.3f})"
        )

    if n_pos == 0 or n_neg == 0:
        print(f"Sign-balanced sampler: degenerate split (pos={n_pos}, neg={n_neg}), use uniform weights.")
        return WeightedRandomSampler([1.0] * n, num_samples=n, replacement=True)

    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    w_zero = 0.5 * (w_pos + w_neg) if n_zero > 0 else 0.0
    weights = np.where(signs > 0, w_pos, np.where(signs < 0, w_neg, w_zero)).astype(np.float32)

    print(f"Sign-balanced weights: w_pos={w_pos:.4f}, w_neg={w_neg:.4f}, zero_w={w_zero:.4f}")
    return WeightedRandomSampler(weights.tolist(), num_samples=n, replacement=True)


def baseline_last_mse(loader: DataLoader) -> float:
    total, cnt = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_grid, _, y_float, _ = unpack_batch(batch)
            last = x_grid[:, -1, 0].float()
            y_abs = y_float.float()
            total += ((last - y_abs) ** 2).sum().item()
            cnt += y_abs.numel()
    return float(total / cnt) if cnt > 0 else 0.0

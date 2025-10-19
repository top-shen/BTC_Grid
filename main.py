# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset import GridKlineDataset
from model import GridTransformer, PatchTST
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributions as D


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="./BTCUSDT_5m.csv")
parser.add_argument(
    "--model",
    type=str,
    default="patchtst",
    choices=["grid", "patchtst"],
    help="which model to use",
)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--bins", type=int, default=60)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-3)

args = parser.parse_args()

NUM_BINS = int(args.bins)  # 用于构建输入 tokens 和统计/可视化

# 最佳模型的保存路径（包含模型类型，避免不同模型互相覆盖）
BEST_CKPT_PATH = f"best_{args.model}.pt"

if args.model == "grid":
    # 使用 GridTransformer 的内置默认结构
    model = GridTransformer(vocab_size=NUM_BINS).to(device)
else:  # patchtst
    # 使用 PatchTST 的内置默认结构
    model = PatchTST(vocab_size=NUM_BINS).to(device)


# 使用分组的 AdamW：对 embedding / norm / bias 取消权重衰减，其他权重保留
decay_params = []
no_decay_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    n = name.lower()
    # 不做权重衰减的典型对象：bias、LayerNorm/Norm 层、各类 embedding
    if n.endswith(".bias") or "norm" in n or "embed" in n or p.ndim < 2:
        no_decay_params.append(p)
    else:
        decay_params.append(p)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=args.lr,
)

# ---- 诊断信息：模型参数量与优化器配置 ----
# 统计参数量（总计与可训练）
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model:", model.__class__.__name__)
print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")

# 打印优化器配置（以第一个参数组为例）
pg0 = optimizer.param_groups[0] if optimizer.param_groups else {}
opt_cfg = {
    "name": optimizer.__class__.__name__,
    "lr": pg0.get("lr"),
    "weight_decay": pg0.get("weight_decay"),
}
print("Optimizer config:", opt_cfg)

df = pd.read_csv(args.csv)

# 训练/测试划分（时间序列：不要打乱）
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# 构建数据集与数据加载器
train_dataset = GridKlineDataset(train_df, num_bins=NUM_BINS)
test_dataset = GridKlineDataset(test_df, num_bins=NUM_BINS)

train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

# Helper to extract last main-channel grid token per batch


def last_main_grid(batch_x: torch.Tensor) -> torch.Tensor:
    # 固定输入为 [B,T,4]，直接取主通道最后一格
    assert batch_x.dim() == 3 and batch_x.size(-1) == 4, "expect x shape [B,T,4]"
    return batch_x[:, -1, 0].float()


# Delta 分布可视化与统计（目标空间：y_abs - last）
def compute_and_plot_delta_distribution(loader: DataLoader, name: str, out_path: str):
    """Collect deltas=y_abs-last and plot histogram. Returns basic stats dict."""
    import numpy as np
    import matplotlib.pyplot as plt

    deltas = []
    with torch.no_grad():
        for x, y_float, _ in loader:
            last = last_main_grid(x)
            deltas.append((y_float.float() - last).numpy())
    if not deltas:
        print(f"[{name}] no data for delta distribution")
        return {"count": 0}
    deltas = np.concatenate(deltas, axis=0)

    # 基本统计信息
    q = np.percentile(deltas, [5, 25, 50, 75, 95])
    frac_1 = np.mean((deltas >= -1.0) & (deltas <= 1.0))
    frac_2 = np.mean((deltas >= -2.0) & (deltas <= 2.0))
    frac_5 = np.mean((deltas >= -5.0) & (deltas <= 5.0))
    print(
        f"[{name}] delta stats: count={deltas.size}, min={deltas.min():.2f}, max={deltas.max():.2f}, "
        f"mean={deltas.mean():.3f}, median={q[2]:.3f}, p5/25/75/95={q[0]:.2f}/{q[1]:.2f}/{q[3]:.2f}/{q[4]:.2f}, "
        f"|d|<=1:{frac_1:.3f}, |d|<=2:{frac_2:.3f}, |d|<=5:{frac_5:.3f}"
    )

    # 直方图（按 1 格为一个bin）
    bins = np.arange(-40.5, 40.5 + 1.0, 1.0)
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


# 在训练前做一次 delta 分布检查（训练集返回尺度统计）
train_stats = compute_and_plot_delta_distribution(
    train_loader, "train", "delta_distribution_train.png"
)
compute_and_plot_delta_distribution(test_loader, "test", "delta_distribution_test.png")

"""
训练目标：相对位移（delta）= y_abs - last_main_grid
损失：Student-t 负对数似然（异方差、重尾友好）

模型头输出 [mu, log_sigma]；其中 sigma = softplus(log_sigma) + eps
mu 表示预测的 delta 均值，最终需要绝对坐标时可用 mu + last。
"""

# 目标空间的对称截断，避免极端值对训练不稳定的影响
DELTA_MIN = -40.0
DELTA_MAX = 40.0

# Student-t 自由度（3~8 常用）。若要更“高斯”，增大；更重尾，减小。
STUDENT_DF = 5.0


# 评估仅保留常见指标（NLL/MAE/RMSE），不再使用 t 分位覆盖率相关指标

scales = (
    train_stats if train_stats.get("count", 0) > 0 else {"q90_abs": 10.0, "std": 5.0}
)
Q90_ABS = max(1.0, float(scales["q90_abs"]))  # 避免过小
SIGMA_PRIOR = max(1.0, float(scales["std"]))  # 以总体 std 作为先验尺度
print(f"Student-t scales: q90_abs={Q90_ABS:.3f}, sigma_prior(std)={SIGMA_PRIOR:.3f}")
# 预先构建 sigma 先验张量，避免训练循环中重复构造
SIGMA_PRIOR_T = torch.tensor(SIGMA_PRIOR, device=device, dtype=torch.float32)

# ---- 构建按 |delta| 分桶的加权采样器（用于训练） ----
print("Building bucketed sampling weights (|delta| buckets)...")


def build_bucketed_sampler(dataset, batch):
    import numpy as np

    tmp_loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    abs_values = []
    with torch.no_grad():
        for x, y_float, _ in tmp_loader:
            last = last_main_grid(x)
            abs_values.append(torch.abs(y_float.float() - last).numpy())
    if not abs_values:
        return None
    abs_values = np.concatenate(abs_values).astype("float64")
    q50, q80, q90 = np.percentile(abs_values, [50, 80, 90])
    print(f"Bucket thresholds: q50={q50:.3f}, q80={q80:.3f}, q90={q90:.3f}")

    def _w(a):
        return 1.0 if a <= q50 else 2.0 if a <= q80 else 3.0 if a <= q90 else 4.0

    weights_arr = np.array([_w(a) for a in abs_values], dtype=np.float32)
    total = len(abs_values)
    print(
        f"Bucket counts: small={(abs_values<=q50).sum()} ({(abs_values<=q50).sum()/total:.3f}), "
        f"mid={((abs_values>q50)&(abs_values<=q80)).sum()} ({((abs_values>q50)&(abs_values<=q80)).sum()/total:.3f}), "
        f"large={((abs_values>q80)&(abs_values<=q90)).sum()} ({((abs_values>q80)&(abs_values<=q90)).sum()/total:.3f}), "
        f"extreme={(abs_values>q90).sum()} ({(abs_values>q90).sum()/total:.3f})"
    )
    return WeightedRandomSampler(
        weights_arr.tolist(), num_samples=len(weights_arr), replacement=True
    )


sampler = build_bucketed_sampler(train_dataset, args.batch)
if sampler is not None:
    train_loader = DataLoader(train_dataset, batch_size=args.batch, sampler=sampler)
    print("Train loader switched to WeightedRandomSampler.")
else:
    print(
        "No training samples to build bucketed sampler; keep original shuffle loader."
    )


SIGMA_REG_LAMBDA = 5e-3  # log-sigma 正则强度
best_val = float("inf")
no_improve_count = 0
patience = 1
num_epochs = int(args.epochs)
for epoch in range(num_epochs):
    # 训练
    model.train()
    total_loss = 0.0

    pbar = tqdm(
        enumerate(train_loader, 1),
        total=len(train_loader),
        desc=f"Train {epoch + 1}/{num_epochs}",
        ncols=100,
    )
    for step, (x, y_float, y_cls) in pbar:
        x = x.to(device)
        y_abs = y_float.to(device)
        # Relative displacement target (delta)
        last = last_main_grid(x).to(device)
        y = (y_abs - last).clamp(min=DELTA_MIN, max=DELTA_MAX)
        optimizer.zero_grad()
        # 预测 [mu, log_sigma]，在 delta 空间下做 Student-t NLL
        out = model(x)  # [B, 2]
        mu, log_s = out[:, 0], out[:, 1]
        s = F.softplus(log_s) + 1e-3
        dist = D.StudentT(df=STUDENT_DF, loc=mu, scale=s)
        # 基础 NLL（不再做基于 |delta| 的样本加权，简化为均值）
        nll = -(dist.log_prob(y))  # [B]
        loss_nll = nll.mean()
        # sigma 正则：
        reg_sigma = (torch.log(s) - torch.log(SIGMA_PRIOR_T)) ** 2
        loss = loss_nll + SIGMA_REG_LAMBDA * reg_sigma.mean()
        loss.backward()
        # 梯度裁剪：限制全局范数，抑制偶发梯度尖峰
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Epoch {epoch + 1}: train_nll = {total_loss / len(train_loader):.4f}")

    # 验证（按 batch 平均，不做样本加权），只保留常见指标
    model.eval()
    val_nll = 0.0
    val_abs_mae = 0.0
    val_abs_mse = 0.0
    with torch.no_grad():
        vbar = tqdm(test_loader, desc=f"Val {epoch + 1}/{num_epochs}", ncols=100)
        for x, y_float, _ in vbar:
            x = x.to(device)
            y_abs = y_float.to(device)
            last = last_main_grid(x).to(device)
            y_tgt = (y_abs - last).clamp(min=DELTA_MIN, max=DELTA_MAX)
            out = model(x)
            mu, log_s = out[:, 0], out[:, 1]
            s = F.softplus(log_s) + 1e-3
            dist = D.StudentT(df=STUDENT_DF, loc=mu, scale=s)
            batch_nll = (-(dist.log_prob(y_tgt))).mean().item()
            mu_abs = (mu + last).clamp(0.0, float(NUM_BINS))
            mae = torch.mean(torch.abs(mu_abs - y_abs)).item()
            mse = torch.mean((mu_abs - y_abs) ** 2).item()
            val_nll += batch_nll
            val_abs_mae += mae
            val_abs_mse += mse
            vbar.set_postfix(nll=f"{batch_nll:.4f}")
    val_nll /= len(test_loader)
    val_abs_mae /= len(test_loader)
    val_abs_rmse = float(val_abs_mse / len(test_loader)) ** 0.5
    print(
        f"Epoch {epoch + 1}: val_nll = {val_nll:.4f}, "
        f"val_mae_abs = {val_abs_mae:.4f}, val_rmse_abs = {val_abs_rmse:.4f}"
    )

    # 提前停止：按 NLL（越小越好）
    if val_nll < best_val:
        best_val = val_nll
        no_improve_count = 0
        # 保存当前最佳验证指标对应的模型
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_nll": float(val_nll),
                "model_type": args.model,
                "num_bins": NUM_BINS,
            },
            BEST_CKPT_PATH,
        )
        print(f"Saved new best model to {BEST_CKPT_PATH} (val_nll={val_nll:.4f})")
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(
                f"Early stop at epoch {epoch + 1}: val NLL did not improve for {patience} epochs"
            )
            break

"""可视化与基线评估"""


def collect_preds(model, loader: DataLoader, num_bins: int, return_last: bool = False):
    preds, reals, lasts = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y_float, _ in loader:
            x = x.to(device)
            y_abs = y_float.to(device)
            last = last_main_grid(x).to(device)
            out = model(x)
            mu = out[:, 0]
            y_hat_abs = (mu + last).clamp(0.0, float(num_bins))
            preds.extend(y_hat_abs.cpu().numpy())
            reals.extend(y_abs.cpu().numpy())
            if return_last:
                lasts.extend(x[:, -1, 0].cpu().numpy())
    if return_last:
        return (
            np.array(preds, dtype=np.float32),
            np.array(reals, dtype=np.float32),
            np.array(lasts, dtype=np.float32),
        )
    else:
        return (
            np.array(preds, dtype=np.float32),
            np.array(reals, dtype=np.float32),
        )


# 在测试集上评估并可视化（使用 mu 作为预测值）
all_preds, all_reals = collect_preds(model, test_loader, NUM_BINS)

# 可视化 1：预测 vs 实际 散点图
plt.figure(figsize=(8, 8))
plt.scatter(all_reals, all_preds, alpha=0.5)
mn, mx = float(min(all_reals.min(), all_preds.min())), float(
    max(all_reals.max(), all_preds.max())
)
plt.plot([mn, mx], [mn, mx], "r--")
plt.title("Predicted vs. Actual Grid (regression)")
plt.xlabel("Actual Grid (float)")
plt.ylabel("Predicted Grid (float)")
plt.grid(True)
plt.savefig("scatter_preds_vs_actuals.png")
plt.close()

# 可视化 2：混淆矩阵（将连续值四舍五入并裁剪到 [0,59]）
all_preds_cls = np.clip(np.floor(all_preds + 0.5).astype(np.int64), 0, NUM_BINS - 1)
all_reals_cls = np.clip(np.floor(all_reals + 0.5).astype(np.int64), 0, NUM_BINS - 1)
conf_mat = confusion_matrix(all_reals_cls, all_preds_cls)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_mat, annot=False, fmt="d")
plt.title("Confusion Matrix (rounded)")
plt.xlabel("Predicted Grid (rounded)")
plt.ylabel("Actual Grid (rounded)")
plt.savefig("confusion_matrix.png")
plt.close()

mse = float(np.mean((all_preds - all_reals) ** 2))
print("\nVisualizations saved as scatter_preds_vs_actuals.png and confusion_matrix.png")
print(f"Test MSE: {mse:.4f}")
print("First 10 Actual Grids:", all_reals[:10])
print("First 10 Predicted Grids:", all_preds[:10])

# 额外可视化：最后2000根K线的预测结果与真实值
last_N = 2000
last_df = df.iloc[-last_N:].copy()
last_dataset = GridKlineDataset(last_df, num_bins=NUM_BINS)
last_loader = DataLoader(last_dataset, batch_size=256, shuffle=False)
last_preds, last_reals, last_grids = collect_preds(
    model, last_loader, NUM_BINS, return_last=True
)

plt.figure(figsize=(12, 4))
plt.plot(last_reals, label="Actual", linewidth=1.2)
plt.plot(last_preds, label="Predicted", linewidth=1.2)
plt.plot(last_grids, label="Last grids", linewidth=1.0, color="tab:green")
plt.title(f"Last {last_N} K-lines: Predicted vs Actual + Last Grids")
plt.xlabel("Index")
plt.ylabel("Grid (float)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("last_2000_pred_vs_actual.png")
plt.close()


# 计算并打印“最后一格”持久性基线（测试集，绝对空间，仅 MSE）
baseline_mse = 0.0
cnt = 0
with torch.no_grad():
    for x, y_float, _ in test_loader:
        last = x[:, -1, 0].float()
        y_abs = y_float.float()
        baseline_mse += ((last - y_abs) ** 2).sum().item()
        cnt += y_abs.numel()
if cnt > 0:
    baseline_mse /= cnt
    print(f"Baseline (last-grid) on test: MSE={baseline_mse:.4f}")

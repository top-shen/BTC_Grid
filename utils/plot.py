# -*- coding: utf-8 -*-
from typing import Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def scatter_preds_vs_actuals(preds: np.ndarray, reals: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(8, 8))
    plt.scatter(reals, preds, alpha=0.5)
    mn = float(min(reals.min(), preds.min()))
    mx = float(max(reals.max(), preds.max()))
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.title("Predicted vs. Actual Grid (regression)")
    plt.xlabel("Actual Grid (float)")
    plt.ylabel("Predicted Grid (float)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def confusion_matrix_plot(preds: np.ndarray, reals: np.ndarray, num_bins: int, out_path: str) -> None:
    preds_cls = np.clip(np.floor(preds + 0.5).astype(np.int64), 0, num_bins - 1)
    reals_cls = np.clip(np.floor(reals + 0.5).astype(np.int64), 0, num_bins - 1)
    conf_mat = confusion_matrix(reals_cls, preds_cls)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_mat, annot=False, fmt="d")
    plt.title("Confusion Matrix (rounded)")
    plt.xlabel("Predicted Grid (rounded)")
    plt.ylabel("Actual Grid (rounded)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_last_curves(reals: np.ndarray, preds: np.ndarray, lasts: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(reals, label="Actual", linewidth=1.2)
    plt.plot(preds, label="Predicted", linewidth=1.2)
    plt.plot(lasts, label="Last grids", linewidth=1.0, color="tab:green")
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Grid (float)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


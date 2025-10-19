# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GridKlineDataset(Dataset):
    """
    多通道网格化数据集。

    - 主通道：历史K线的典型价格 mid = (high+low+close)/3，映射到 21..40 的20格。
    - 额外通道：ma5、ma10、ma20（基于 close 的简单移动均线），也映射到同一窗口网格 21..40。
    - 目标：未来收盘价（t+pred_len-1）按 L0..L60 的 60 格得到连续坐标 y_float ∈ [0,60]，并提供离散 y_cls ∈ [0,59]。

    输出：
      x: [T, 4] 的 long tokens，通道顺序为 [main, ma5, ma10, ma20]
      y_float: float32，连续坐标
      y_cls: long，离散格索引
    """

    def __init__(self, df, window_size=300, pred_len=50, num_bins=60):
        self.close = df["close"].values.astype(np.float64)
        self.high = df["high"].values.astype(np.float64)
        self.low = df["low"].values.astype(np.float64)
        self.window_size = int(window_size)
        self.pred_len = int(pred_len)
        self.num_bins = int(num_bins)

        # 预计算移动均线（与原始索引对齐）
        s_close = pd.Series(self.close)
        ma5 = s_close.rolling(window=5, min_periods=5).mean().values
        ma10 = s_close.rolling(window=10, min_periods=10).mean().values
        ma20 = s_close.rolling(window=20, min_periods=20).mean().values
        self.ma5 = ma5
        self.ma10 = ma10
        self.ma20 = ma20

        # 起始索引需要保证 ma20 全部有定义
        start_i = 20 - 1  # 19

        self.samples = []  # Each sample: (grid_hist_tokens [T,4], y_float, y_cls)
        L = len(self.close)
        for i in range(start_i, L - self.window_size - self.pred_len):
            # 窗口区间上下界：使用窗口内历史 K 线的最高/最低
            high_w = float(np.max(self.high[i : i + self.window_size]))
            low_w = float(np.min(self.low[i : i + self.window_size]))
            rng = high_w - low_w
            if rng <= 1e-12:
                rng = 1e-12  # 防止区间为 0
            delta = rng / 20.0  # L20..L40 之间 20 个格子

            # 主通道：典型价格 -> 网格 21..40
            mid_hist = (
                self.high[i : i + self.window_size]
                + self.low[i : i + self.window_size]
                + self.close[i : i + self.window_size]
            ) / 3.0
            idx_main = np.floor((mid_hist - low_w) / delta).astype(int)
            idx_main = np.clip(idx_main, 0, 19)  # 0..19 -> 21..40
            grid_main = idx_main + 21

            # 额外通道：ma5/ma10/ma20 使用同一窗口网格边界映射到 21..40
            ma5_win = self.ma5[i : i + self.window_size]
            ma10_win = self.ma10[i : i + self.window_size]
            ma20_win = self.ma20[i : i + self.window_size]
            # 起始 i>=19 且 rolling(min_periods=5/10/20) 已确保无 NaN
            assert not (
                np.isnan(ma5_win).any()
                or np.isnan(ma10_win).any()
                or np.isnan(ma20_win).any()
            ), "MA windows contain NaN; please check input data"

            idx_ma5 = np.floor((ma5_win - low_w) / delta).astype(int)
            idx_ma10 = np.floor((ma10_win - low_w) / delta).astype(int)
            idx_ma20 = np.floor((ma20_win - low_w) / delta).astype(int)
            idx_ma5 = np.clip(idx_ma5, 0, 19)
            idx_ma10 = np.clip(idx_ma10, 0, 19)
            idx_ma20 = np.clip(idx_ma20, 0, 19)
            grid_ma5 = idx_ma5 + 21
            grid_ma10 = idx_ma10 + 21
            grid_ma20 = idx_ma20 + 21

            grid_hist = np.stack([grid_main, grid_ma5, grid_ma10, grid_ma20], axis=-1)

            # 预测目标：未来收盘价 -> L0..L60 连续坐标
            l0 = low_w - 20.0 * delta
            close_future = self.close[i + self.window_size + self.pred_len - 1]
            idx_raw = (close_future - l0) / delta  # 连续坐标，可越界
            y_float = float(np.clip(idx_raw, 0.0, float(self.num_bins)))  # [0,60]
            idx_future = int(np.floor(idx_raw))
            y_cls = int(np.clip(idx_future, 0, self.num_bins - 1))  # [0,59]

            self.samples.append((grid_hist.astype(np.int64), y_float, y_cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y_float, y_cls = self.samples[idx]
        x = torch.tensor(x, dtype=torch.long)
        y_float = torch.tensor(y_float, dtype=torch.float32)
        y_cls = torch.tensor(y_cls, dtype=torch.long)
        return x, y_float, y_cls

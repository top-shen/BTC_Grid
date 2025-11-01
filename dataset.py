# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GridKlineDataset(Dataset):
    """
    多通道网格化数据集。

    - 主通道：历史K线的典型价格 mid = (high+low+close)/3，映射到 21..40 的20格。
    - 额外通道（同一窗口网格 21..40）：
        * ma5、ma10、ma20（基于 close 的简单移动均线）
        * vwap5、vwap10、vwap20（滚动成交量加权价格，价格使用典型价 mid）
    - 目标：未来收盘价（t+pred_len-1）按 L0..L60 的 60 格得到连续坐标 y_float ∈ [0,60]，并提供离散 y_cls ∈ [0,59]。

    输出：
      x: [T, 7] 的 long tokens，通道顺序为
         [main, ma5, ma10, ma20, vwap5, vwap10, vwap20]
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

        # 成交量与典型价，用于 VWAP（滚动窗口）
        # volume 列名兜底：优先使用 'volume'，否则尝试 'vol' 或 'amount'
        vol_col = None
        for c in ["volume", "vol", "Vol", "amount", "Amount"]:
            if c in df.columns:
                vol_col = c
                break
        if vol_col is None:
            raise KeyError(
                "VWAP requires a volume-like column; expected one of ['volume','vol','amount'] in df"
            )
        self.volume = df[vol_col].values.astype(np.float64)
        # 典型价序列：与主通道一致
        self.mid_all = (self.high + self.low + self.close) / 3.0

        s_vol = pd.Series(self.volume)
        s_pv = pd.Series(self.mid_all * self.volume)

        def _rolling_vwap(n: int) -> np.ndarray:
            # vwap_n = sum(price*vol, n)/sum(vol, n)；当分母极小则返回 NaN
            num = s_pv.rolling(window=n, min_periods=n).sum()
            den = s_vol.rolling(window=n, min_periods=n).sum()
            v = num / den.replace(0.0, np.nan)
            return np.array(v.values)

        self.vwap5 = _rolling_vwap(5)
        self.vwap10 = _rolling_vwap(10)
        self.vwap20 = _rolling_vwap(20)
        # 通道数量（主通道 + 3MA + 3VWAP）
        self.num_channels = 7

        # 起始索引需要保证 ma20 全部有定义
        start_i = 20 - 1  # 19，保证 ma20/vwap20 已定义

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
            mid_hist = self.mid_all[i : i + self.window_size]
            idx_main = np.floor((mid_hist - low_w) / delta).astype(int)
            idx_main = np.clip(idx_main, 0, 19)  # 0..19 -> 21..40
            grid_main = idx_main + 21

            # 额外通道：ma5/ma10/ma20 使用同一窗口网格边界映射到 21..40
            ma5_win = np.asarray(self.ma5[i : i + self.window_size])
            ma10_win = np.asarray(self.ma10[i : i + self.window_size])
            ma20_win = np.asarray(self.ma20[i : i + self.window_size])
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

            # VWAP 通道：vwap5/10/20 映射到同一窗口 21..40
            vwap5_win = np.asarray(self.vwap5[i : i + self.window_size])
            vwap10_win = np.asarray(self.vwap10[i : i + self.window_size])
            vwap20_win = np.asarray(self.vwap20[i : i + self.window_size])
            # 有些窗口内可能出现零成交量，导致滚动 VWAP 的分母为 0，从而产生 NaN。
            # 根据需求，遇到这类样本直接剔除（跳过），而不是中断训练。
            if (
                np.isnan(vwap5_win).any()
                or np.isnan(vwap10_win).any()
                or np.isnan(vwap20_win).any()
            ):
                # Skip this sample; it crosses a zero-volume area that makes VWAP undefined.
                continue

            idx_v5 = np.floor((vwap5_win - low_w) / delta).astype(int)
            idx_v10 = np.floor((vwap10_win - low_w) / delta).astype(int)
            idx_v20 = np.floor((vwap20_win - low_w) / delta).astype(int)
            idx_v5 = np.clip(idx_v5, 0, 19)
            idx_v10 = np.clip(idx_v10, 0, 19)
            idx_v20 = np.clip(idx_v20, 0, 19)
            grid_v5 = idx_v5 + 21
            grid_v10 = idx_v10 + 21
            grid_v20 = idx_v20 + 21

            grid_hist = np.stack(
                [
                    grid_main,
                    grid_ma5,
                    grid_ma10,
                    grid_ma20,
                    grid_v5,
                    grid_v10,
                    grid_v20,
                ],
                axis=-1,
            )

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

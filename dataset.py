# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GridKlineDataset(Dataset):
    """Build delta-grid samples for the baseline and tokenizer ablations.

    Baseline mode output:
      x_grid:  [T, 7] long tokens
      y_float: scalar float grid target in [0, num_bins]
      y_cls:   scalar long class target in [0, num_bins-1]

    Tokenizer mode output (`return_continuous=True`):
      x_grid:  [T, 7] long tokens kept for the baseline task definition / backtest path
      x_tok:   [T, F_tok] float tokenizer input sequence
      y_float: scalar float grid target
      y_cls:   scalar long class target
    """

    def __init__(
        self,
        df,
        window_size=300,
        pred_len=50,
        num_bins=60,
        return_continuous: bool = False,
        tokenizer_input_mode: str = "grid_cont",
    ):
        self.open = df["open"].values.astype(np.float64) if "open" in df.columns else df["close"].values.astype(np.float64)
        self.close = df["close"].values.astype(np.float64)
        self.high = df["high"].values.astype(np.float64)
        self.low = df["low"].values.astype(np.float64)
        self.window_size = int(window_size)
        self.pred_len = int(pred_len)
        self.num_bins = int(num_bins)
        self.return_continuous = bool(return_continuous)
        self.tokenizer_input_mode = str(tokenizer_input_mode)
        if self.tokenizer_input_mode not in {"grid_cont", "raw_ohlcv"}:
            raise ValueError(f"Unsupported tokenizer_input_mode={self.tokenizer_input_mode}")

        s_close = pd.Series(self.close)
        self.ma5 = s_close.rolling(window=5, min_periods=5).mean().values
        self.ma10 = s_close.rolling(window=10, min_periods=10).mean().values
        self.ma20 = s_close.rolling(window=20, min_periods=20).mean().values

        vol_col = None
        for c in ["volume", "vol", "Vol", "amount", "Amount"]:
            if c in df.columns:
                vol_col = c
                break
        if vol_col is None:
            raise KeyError("VWAP requires a volume-like column; expected one of ['volume','vol','amount'] in df")
        self.volume = df[vol_col].values.astype(np.float64)
        self.mid_all = (self.high + self.low + self.close) / 3.0

        s_vol = pd.Series(self.volume)
        s_pv = pd.Series(self.mid_all * self.volume)

        def _rolling_vwap(n: int) -> np.ndarray:
            num = s_pv.rolling(window=n, min_periods=n).sum()
            den = s_vol.rolling(window=n, min_periods=n).sum()
            return np.array((num / den.replace(0.0, np.nan)).values)

        self.vwap5 = _rolling_vwap(5)
        self.vwap10 = _rolling_vwap(10)
        self.vwap20 = _rolling_vwap(20)

        self.num_channels = 7
        self.feature_dim = 7 if self.tokenizer_input_mode == "grid_cont" else 5

        start_i = 20 - 1
        self.samples = []
        total_len = len(self.close)

        for i in range(start_i, total_len - self.window_size - self.pred_len):
            high_w = float(np.max(self.high[i : i + self.window_size]))
            low_w = float(np.min(self.low[i : i + self.window_size]))
            rng = max(high_w - low_w, 1e-12)
            delta = rng / 20.0

            mid_hist = self.mid_all[i : i + self.window_size]
            ma5_win = np.asarray(self.ma5[i : i + self.window_size])
            ma10_win = np.asarray(self.ma10[i : i + self.window_size])
            ma20_win = np.asarray(self.ma20[i : i + self.window_size])
            if np.isnan(ma5_win).any() or np.isnan(ma10_win).any() or np.isnan(ma20_win).any():
                continue

            vwap5_win = np.asarray(self.vwap5[i : i + self.window_size])
            vwap10_win = np.asarray(self.vwap10[i : i + self.window_size])
            vwap20_win = np.asarray(self.vwap20[i : i + self.window_size])
            if np.isnan(vwap5_win).any() or np.isnan(vwap10_win).any() or np.isnan(vwap20_win).any():
                continue

            def _grid_coords(values: np.ndarray) -> np.ndarray:
                coords = (values - low_w) / delta
                coords = np.clip(coords, 0.0, 19.999)
                return coords + 21.0

            cont_main = _grid_coords(mid_hist)
            cont_ma5 = _grid_coords(ma5_win)
            cont_ma10 = _grid_coords(ma10_win)
            cont_ma20 = _grid_coords(ma20_win)
            cont_v5 = _grid_coords(vwap5_win)
            cont_v10 = _grid_coords(vwap10_win)
            cont_v20 = _grid_coords(vwap20_win)

            grid_hist = np.stack(
                [
                    np.floor(cont_main).astype(np.int64),
                    np.floor(cont_ma5).astype(np.int64),
                    np.floor(cont_ma10).astype(np.int64),
                    np.floor(cont_ma20).astype(np.int64),
                    np.floor(cont_v5).astype(np.int64),
                    np.floor(cont_v10).astype(np.int64),
                    np.floor(cont_v20).astype(np.int64),
                ],
                axis=-1,
            )
            grid_cont = np.stack(
                [cont_main, cont_ma5, cont_ma10, cont_ma20, cont_v5, cont_v10, cont_v20],
                axis=-1,
            ).astype(np.float32)

            open_win = self.open[i : i + self.window_size]
            high_win = self.high[i : i + self.window_size]
            low_win = self.low[i : i + self.window_size]
            close_win = self.close[i : i + self.window_size]
            volume_win = self.volume[i : i + self.window_size]
            price_ref = float(max(abs(close_win[0]), 1e-6))
            price_stack = np.stack([open_win, high_win, low_win, close_win], axis=-1)
            price_stack = price_stack / price_ref - 1.0
            vol_log = np.log1p(np.maximum(volume_win, 0.0))
            vol_std = float(max(vol_log.std(), 1e-6))
            vol_norm = ((vol_log - vol_log.mean()) / vol_std)[:, None]
            raw_ohlcv = np.concatenate([price_stack, vol_norm], axis=-1).astype(np.float32)

            l0 = low_w - 20.0 * delta
            close_future = self.close[i + self.window_size + self.pred_len - 1]
            idx_raw = (close_future - l0) / delta
            y_float = float(np.clip(idx_raw, 0.0, float(self.num_bins)))
            y_cls = int(np.clip(np.floor(idx_raw), 0, self.num_bins - 1))

            if self.return_continuous:
                x_tok = grid_cont if self.tokenizer_input_mode == "grid_cont" else raw_ohlcv
                self.samples.append((grid_hist, x_tok, y_float, y_cls))
            else:
                self.samples.append((grid_hist, y_float, y_cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if self.return_continuous:
            x_grid, x_tok, y_float, y_cls = item
            return (
                torch.tensor(x_grid, dtype=torch.long),
                torch.tensor(x_tok, dtype=torch.float32),
                torch.tensor(y_float, dtype=torch.float32),
                torch.tensor(y_cls, dtype=torch.long),
            )

        x_grid, y_float, y_cls = item
        return (
            torch.tensor(x_grid, dtype=torch.long),
            torch.tensor(y_float, dtype=torch.float32),
            torch.tensor(y_cls, dtype=torch.long),
        )

# -*- coding: utf-8 -*-
"""
Quantile trainer: joint training of multiple quantiles (e.g., 0.1/0.5/0.9)
with a single model that has a shared trunk and multiple linear heads.

Target (this version): delta = y_abs - last_main_grid(x)
- Loss: sum of pinball (quantile) losses across all quantiles on delta
  L = sum_q mean( max(q*(d - d_hat_q), (q-1)*(d - d_hat_q)) )
During evaluation and collection, we add `last` back to get absolute grids.
"""
from typing import Dict, Iterable, Tuple
import torch
import torch.nn.functional as F
from .data import last_main_grid


def pinball_loss(y: torch.Tensor, y_hat: torch.Tensor, q: float) -> torch.Tensor:
    """Pinball loss for a single quantile q in (0,1). Returns mean over batch."""
    e = y - y_hat
    return torch.mean(torch.maximum(q * e, (q - 1.0) * e))


class QuantileTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        quantiles: Iterable[float] = (0.1, 0.5, 0.9),
    ) -> None:
        self.model = model
        self.device = device
        self.quantiles = [float(x) for x in quantiles]
        assert len(self.quantiles) >= 2, "need at least two quantiles"

    def _compute_loss(self, out: torch.Tensor, y_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (loss, q_delta_preds)

        - out: [B, Q] (predicted quantiles of delta)
        - y_delta: [B] (target delta)
        """
        assert out.dim() == 2 and out.size(1) == len(self.quantiles), (
            f"model output shape {tuple(out.shape)} does not match quantiles={self.quantiles}"
        )
        losses = []
        for j, q in enumerate(self.quantiles):
            losses.append(pinball_loss(y_delta, out[:, j], q))
        loss = torch.stack(losses).sum()

        return loss, out

    def train_epoch(self, loader, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y_float, _ in loader:
            x = x.to(self.device)
            y_abs = y_float.to(self.device)
            # delta target relative to last main grid token
            last = last_main_grid(x).to(self.device)
            y_delta = y_abs - last
            optimizer.zero_grad()
            out = self.model(x)
            loss, _ = self._compute_loss(out, y_delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
        return float(total_loss / max(1, len(loader)))

    @torch.no_grad()
    def validate(self, loader, num_bins: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        abs_mae = 0.0
        abs_mse = 0.0
        cov_counts = {q: 0.0 for q in self.quantiles}
        cov80 = 0.0  # P(q10 <= y <= q90) if exactly 3 quantiles
        n_batches = 0
        n_obs = 0

        for x, y_float, _ in loader:
            x = x.to(self.device)
            y_abs = y_float.to(self.device)
            last = last_main_grid(x).to(self.device)
            y_delta = y_abs - last
            out = self.model(x)
            loss, q_delta = self._compute_loss(out, y_delta)
            total_loss += float(loss.item())
            n_batches += 1

            # Point metrics on q50 (if present), else median quantile (middle index)
            mid_idx = len(self.quantiles) // 2
            y_hat_abs = (q_delta[:, mid_idx] + last).clamp(0.0, float(num_bins))
            err = y_hat_abs - y_abs
            abs_mae += float(err.abs().sum().item())
            abs_mse += float((err * err).sum().item())
            n_obs += int(y_abs.numel())

            # Quantile coverage calibration
            for j, qq in enumerate(self.quantiles):
                q_abs_j = q_delta[:, j] + last
                cov_counts[qq] += float((y_abs <= q_abs_j).float().sum().item())
            if len(self.quantiles) >= 3:
                q_abs = q_delta + last.unsqueeze(1)
                q_sorted, _ = torch.sort(q_abs, dim=1)
                cov80 += float(((y_abs >= q_sorted[:, 0]) & (y_abs <= q_sorted[:, -1])).float().sum().item())

        mae = abs_mae / max(1, n_obs)
        mse = abs_mse / max(1, n_obs)
        qloss = total_loss / max(1, n_batches)
        report = {"qloss": qloss, "mae": mae, "mse": mse, "count": float(n_obs)}
        for qq, cc in cov_counts.items():
            report[f"cov@{qq}"] = cc / max(1, n_obs)
        if len(self.quantiles) >= 3:
            report["cov@interval"] = cov80 / max(1, n_obs)
        return report

    @torch.no_grad()
    def collect_preds(self, loader, num_bins: int, return_last: bool = False):
        """Return (preds_q50, reals, lasts?) in numpy arrays.

        - For compatibility with downstream plots, return q50 absolute as point prediction.
        """
        preds, reals, lasts = [], [], []
        self.model.eval()
        for x, y_float, _ in loader:
            x = x.to(self.device)
            y_abs = y_float.to(self.device)
            out = self.model(x)
            last = last_main_grid(x).to(self.device)
            mid_idx = len(self.quantiles) // 2
            y_hat_abs = (out[:, mid_idx] + last).clamp(0.0, float(num_bins))
            preds.extend(y_hat_abs.cpu().numpy())
            reals.extend(y_abs.cpu().numpy())
            if return_last:
                lasts.extend(x[:, -1, 0].cpu().numpy())
        import numpy as np
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

    @torch.no_grad()
    def collect_quantiles(self, loader, num_bins: int):
        """Return stacked quantiles [N, Q] and reals [N]."""
        all_q, reals = [], []
        self.model.eval()
        for x, y_float, _ in loader:
            x = x.to(self.device)
            y_abs = y_float.to(self.device)
            last = last_main_grid(x).to(self.device)
            q_abs = (self.model(x) + last.unsqueeze(1)).clamp(0.0, float(num_bins))
            all_q.append(q_abs.cpu())
            reals.append(y_abs.cpu())
        import torch as _t
        return _t.cat(all_q, dim=0).numpy(), _t.cat(reals, dim=0).numpy()

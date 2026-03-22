# -*- coding: utf-8 -*-
"""Quantile trainer for the baseline and tokenizer ablation paths."""
from typing import Dict, Iterable, Tuple
import torch
from .data import last_main_grid, unpack_batch


def pinball_loss(y: torch.Tensor, y_hat: torch.Tensor, q: float) -> torch.Tensor:
    e = y - y_hat
    return torch.mean(torch.maximum(q * e, (q - 1.0) * e))


def forward_quantile_model(model: torch.nn.Module, grid_x: torch.Tensor, tok_x: torch.Tensor = None):
    if tok_x is None:
        output = model(grid_x)
    else:
        output = model(grid_x, tok_x)

    if isinstance(output, dict):
        pred = output["pred"]
        aux = {k: v for k, v in output.items() if k != "pred"}
    else:
        pred = output
        aux = {}
    return pred, aux


class QuantileTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        quantiles: Iterable[float] = (0.1, 0.5, 0.9),
        lambda_vq: float = 1.0,
    ) -> None:
        self.model = model
        self.device = device
        self.quantiles = [float(x) for x in quantiles]
        self.lambda_vq = float(lambda_vq)
        assert len(self.quantiles) >= 2, "need at least two quantiles"

    def _compute_loss(self, out: torch.Tensor, y_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert out.dim() == 2 and out.size(1) == len(self.quantiles), (
            f"model output shape {tuple(out.shape)} does not match quantiles={self.quantiles}"
        )
        losses = [pinball_loss(y_delta, out[:, j], q) for j, q in enumerate(self.quantiles)]
        loss = torch.stack(losses).sum()
        return loss, out

    @staticmethod
    def _scalar(aux: Dict[str, torch.Tensor], key: str, device: torch.device) -> torch.Tensor:
        value = aux.get(key)
        if value is None:
            return torch.zeros((), device=device)
        if not torch.is_tensor(value):
            return torch.tensor(float(value), device=device)
        return value if value.dim() == 0 else value.mean()

    def _empty_stats(self) -> Dict[str, float]:
        return {
            "pred_qloss": 0.0,
            "vq_loss": 0.0,
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "codebook_loss": 0.0,
            "commitment_loss": 0.0,
            "diversity_loss": 0.0,
            "orthogonal_loss": 0.0,
            "used_codes": 0.0,
            "top1_code_freq": 0.0,
        }

    def train_epoch(self, loader, optimizer) -> Dict[str, float]:
        self.model.train()
        totals = self._empty_stats()

        for batch in loader:
            grid_x, tok_x, y_float, _ = unpack_batch(batch)
            grid_x = grid_x.to(self.device)
            tok_x = tok_x.to(self.device) if tok_x is not None else None
            y_abs = y_float.to(self.device)
            last = last_main_grid(grid_x).to(self.device)
            y_delta = y_abs - last

            optimizer.zero_grad()
            pred, aux = forward_quantile_model(self.model, grid_x, tok_x)
            pred_loss, _ = self._compute_loss(pred, y_delta)
            vq_loss = self._scalar(aux, "vq_loss", self.device)
            total_loss = pred_loss + self.lambda_vq * vq_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            totals["pred_qloss"] += float(pred_loss.item())
            totals["vq_loss"] += float(vq_loss.item())
            totals["total_loss"] += float(total_loss.item())
            for key in [
                "recon_loss",
                "codebook_loss",
                "commitment_loss",
                "diversity_loss",
                "orthogonal_loss",
                "used_codes",
                "top1_code_freq",
            ]:
                totals[key] += float(self._scalar(aux, key, self.device).item())

        denom = max(1, len(loader))
        return {key: value / denom for key, value in totals.items()}

    @torch.no_grad()
    def validate(self, loader, num_bins: int) -> Dict[str, float]:
        self.model.eval()
        totals = self._empty_stats()
        abs_mae = 0.0
        abs_mse = 0.0
        cov_counts = {q: 0.0 for q in self.quantiles}
        cov80 = 0.0
        n_batches = 0
        n_obs = 0

        for batch in loader:
            grid_x, tok_x, y_float, _ = unpack_batch(batch)
            grid_x = grid_x.to(self.device)
            tok_x = tok_x.to(self.device) if tok_x is not None else None
            y_abs = y_float.to(self.device)
            last = last_main_grid(grid_x).to(self.device)
            y_delta = y_abs - last

            pred, aux = forward_quantile_model(self.model, grid_x, tok_x)
            pred_loss, q_delta = self._compute_loss(pred, y_delta)
            vq_loss = self._scalar(aux, "vq_loss", self.device)
            totals["pred_qloss"] += float(pred_loss.item())
            totals["vq_loss"] += float(vq_loss.item())
            totals["total_loss"] += float((pred_loss + self.lambda_vq * vq_loss).item())
            for key in [
                "recon_loss",
                "codebook_loss",
                "commitment_loss",
                "diversity_loss",
                "orthogonal_loss",
                "used_codes",
                "top1_code_freq",
            ]:
                totals[key] += float(self._scalar(aux, key, self.device).item())
            n_batches += 1

            mid_idx = len(self.quantiles) // 2
            y_hat_abs = (q_delta[:, mid_idx] + last).clamp(0.0, float(num_bins))
            err = y_hat_abs - y_abs
            abs_mae += float(err.abs().sum().item())
            abs_mse += float((err * err).sum().item())
            n_obs += int(y_abs.numel())

            for j, qq in enumerate(self.quantiles):
                q_abs_j = q_delta[:, j] + last
                cov_counts[qq] += float((y_abs <= q_abs_j).float().sum().item())
            if len(self.quantiles) >= 3:
                q_abs = q_delta + last.unsqueeze(1)
                q_sorted, _ = torch.sort(q_abs, dim=1)
                cov80 += float(((y_abs >= q_sorted[:, 0]) & (y_abs <= q_sorted[:, -1])).float().sum().item())

        mae = abs_mae / max(1, n_obs)
        mse = abs_mse / max(1, n_obs)
        report = {key if key != "pred_qloss" else "qloss": value / max(1, n_batches) for key, value in totals.items()}
        report.update({"mae": mae, "mse": mse, "count": float(n_obs)})
        for qq, cc in cov_counts.items():
            report[f"cov@{qq}"] = cc / max(1, n_obs)
        if len(self.quantiles) >= 3:
            report["cov@interval"] = cov80 / max(1, n_obs)
        return report

    @torch.no_grad()
    def collect_preds(self, loader, num_bins: int, return_last: bool = False):
        preds, reals, lasts = [], [], []
        self.model.eval()
        for batch in loader:
            grid_x, tok_x, y_float, _ = unpack_batch(batch)
            grid_x = grid_x.to(self.device)
            tok_x = tok_x.to(self.device) if tok_x is not None else None
            y_abs = y_float.to(self.device)
            pred, _ = forward_quantile_model(self.model, grid_x, tok_x)
            last = last_main_grid(grid_x).to(self.device)
            mid_idx = len(self.quantiles) // 2
            y_hat_abs = (pred[:, mid_idx] + last).clamp(0.0, float(num_bins))
            preds.extend(y_hat_abs.cpu().numpy())
            reals.extend(y_abs.cpu().numpy())
            if return_last:
                lasts.extend(grid_x[:, -1, 0].cpu().numpy())
        import numpy as np
        if return_last:
            return np.array(preds, dtype=np.float32), np.array(reals, dtype=np.float32), np.array(lasts, dtype=np.float32)
        return np.array(preds, dtype=np.float32), np.array(reals, dtype=np.float32)

    @torch.no_grad()
    def collect_quantiles(self, loader, num_bins: int):
        all_q, reals = [], []
        self.model.eval()
        for batch in loader:
            grid_x, tok_x, y_float, _ = unpack_batch(batch)
            grid_x = grid_x.to(self.device)
            tok_x = tok_x.to(self.device) if tok_x is not None else None
            y_abs = y_float.to(self.device)
            last = last_main_grid(grid_x).to(self.device)
            pred, _ = forward_quantile_model(self.model, grid_x, tok_x)
            q_abs = (pred + last.unsqueeze(1)).clamp(0.0, float(num_bins))
            all_q.append(q_abs.cpu())
            reals.append(y_abs.cpu())
        import torch as _t
        return _t.cat(all_q, dim=0).numpy(), _t.cat(reals, dim=0).numpy()

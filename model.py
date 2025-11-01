# -*- coding: utf-8 -*-
# model.py
from typing import Optional, Sequence
import torch
import torch.nn as nn


def _embed_channels(
    x: torch.Tensor,
    embeds: Sequence[nn.Module],  # accept generic nn.Module sequence to play well with ModuleList
    channel_combine: str,
    channel_proj: Optional[nn.Module],
) -> torch.Tensor:
    """Embed [B,T,C] token tensor with C separate embeddings and combine.

    - channel_combine == 'sum': elementwise sum of per-channel embeddings.
    - channel_combine == 'concat': concat along last dim then project back to d_model.
    """
    assert x.dim() == 3 and x.size(-1) == len(embeds), (
        f"expect x shape [B,T,{len(embeds)}] to match num_channels"
    )
    hs = [emb(x[..., i]) for i, emb in enumerate(embeds)]
    if channel_combine == "concat":
        assert channel_proj is not None, "channel_proj must be set when using concat"
        h = torch.cat(hs, dim=-1)
        h = channel_proj(h)
    else:
        h = hs[0]
        for t in hs[1:]:
            h = h + t
    return h


class PatchTST(nn.Module):
    """
    PatchTST-style encoder (closer to common source impl) for sequence-to-one regression
    on tokenized grids.

    Notes:
    - By default this head now outputs 3 values intended for quantile regression
      (q10, q50, q90). If you need the previous [mu, log_sigma] head for Student-t
      modelling, set `out_dim=2` when constructing the module.
    - The trunk/encoder is unchanged.
    """

    def __init__(
        self,
        vocab_size=60,
        num_channels: int = 7,
        # Moderate defaults: less reduction than lite, still smaller than the original heavy cfg
        d_model=96,
        num_heads=8,
        num_layers=3,
        patch_len=16,
        stride=8,
        max_patches=512,
        ff_dim=768,
        dropout=0.2,
        emb_dropout=0.2,
        head_dropout=0.2,
        pos_dropout=0.2,
        pool="cls",  # "last", "mean", or "cls"
        head_kind="mlp",  # "mlp" or "linear"
        patch_depthwise=True,  # depthwise separable conv for patching
        channel_combine="concat",  # "sum" (fewer params) or "concat" (richer)
        out_dim: int = 3,  # default to 3 quantile outputs: [q10, q50, q90]
    ):
        super().__init__()
        self.pool = pool
        self.max_patches = int(max_patches)
        self.head_kind = head_kind
        self.out_dim = int(out_dim)

        # Token embeddings per channel (e.g., [main, ma5, ma10, ma20, vwap5, vwap10, vwap20])
        self.num_channels = int(num_channels)
        self.embeds = nn.ModuleList(
            [nn.Embedding(vocab_size, d_model) for _ in range(self.num_channels)]
        )
        # Combine channel embeddings; "sum" removes an extra projection layer
        self.channel_combine = channel_combine
        if channel_combine == "concat":
            self.channel_proj = nn.Linear(d_model * self.num_channels, d_model)
        else:
            self.channel_proj = None
        self.emb_drop = nn.Dropout(emb_dropout)

        # Conv1d patching over time: [B, d_model, T] -> [B, d_model, N]
        if patch_depthwise:
            # depthwise separable: fewer params
            self.patch_proj = nn.Sequential(
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=patch_len,
                    stride=stride,
                    groups=d_model,
                ),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            )
        else:
            self.patch_proj = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=patch_len,
                stride=stride,
            )

        # Learnable positional embedding for patches (+1 for CLS) and dropout
        self.pos_embed_patch = nn.Parameter(
            torch.zeros(1, self.max_patches + 1, d_model)
        )
        nn.init.normal_(self.pos_embed_patch, std=0.02)
        self.pos_drop = nn.Dropout(pos_dropout)

        # CLS token and pre-head LayerNorm
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.pre_head_ln = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if head_kind == "linear":
            # 默认输出为 3 维（q10, q50, q90）；如需旧版 [mu, log_sigma]，请构造 out_dim=2
            self.reg_head = nn.Linear(d_model, self.out_dim)
        else:
            self.reg_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(d_model, self.out_dim),  # 通常为 3: [q10, q50, q90]
            )

    def _embed_channels(self, x):
        """x: [B,T,C] long -> [B,T,d_model]"""
        return _embed_channels(
            x, tuple(self.embeds), self.channel_combine, self.channel_proj  # tuple() satisfies Sequence for type-checkers
        )

    def forward(self, x):
        # 多通道输入，形状断言在嵌入函数中完成
        h = self._embed_channels(x)  # [B, T, d_model]
        h = self.emb_drop(h)

        # Conv1d over time axis for patching
        h = h.transpose(1, 2)  # [B, d_model, T]
        p = self.patch_proj(h)  # [B, d_model, N]
        p = p.transpose(1, 2)  # [B, N, d_model]

        # Prepend CLS and add positional embedding + dropout
        B, N, D = p.shape
        cls = self.cls_token.expand(B, -1, -1)  # [B,1,D]
        p = torch.cat([cls, p], dim=1)  # [B, N+1, D]

        # Slice or pad positional embedding to match N+1 length
        need = N + 1
        if need <= self.pos_embed_patch.size(1):
            pos = self.pos_embed_patch[:, :need, :]
        else:
            extra = need - self.pos_embed_patch.size(1)
            pad = torch.zeros(1, extra, D, device=p.device, dtype=p.dtype)
            pos = torch.cat([self.pos_embed_patch, pad], dim=1)
        p = self.pos_drop(p + pos)

        # Encode
        z = self.encoder(p)  # [B, N+1, d_model]

        # Aggregate
        if self.pool == "mean":
            z = z[:, 1:, :].mean(dim=1)  # exclude CLS
        elif self.pool == "cls":
            z = z[:, 0, :]
        else:  # last
            z = z[:, -1, :]

        z = self.pre_head_ln(z)
        out = self.reg_head(z)  # [B, 2]
        return out

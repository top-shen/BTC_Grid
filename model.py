# -*- coding: utf-8 -*-
# model.py
from typing import Optional, Sequence
import torch
import torch.nn as nn


def _embed_channels(
    x: torch.Tensor,
    embeds: Sequence[nn.Module],
    channel_combine: str,
    channel_proj: Optional[nn.Module],
) -> torch.Tensor:
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
    """PatchTST-style predictor for delta-grid quantile regression.

    Baseline path:
      discrete grid tokens [B, T, C] -> embeddings -> optional repatch -> Transformer -> q10/q50/q90

    Tokenizer path:
      embedding sequence [B, N, D] ->
        - repatch: treat embeddings as a sequence and patch again
        - direct_tokens: feed embeddings directly as transformer tokens
    """

    def __init__(
        self,
        vocab_size=60,
        num_channels: int = 7,
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
        pool="cls",
        head_kind="mlp",
        patch_depthwise=True,
        channel_combine="concat",
        out_dim: int = 3,
        token_mode: str = "repatch",
    ):
        super().__init__()
        self.pool = pool
        self.max_patches = int(max_patches)
        self.head_kind = head_kind
        self.out_dim = int(out_dim)
        self.d_model = int(d_model)
        self.token_mode = str(token_mode)
        if self.token_mode not in {"repatch", "direct_tokens"}:
            raise ValueError(f"Unsupported token_mode={self.token_mode}")

        self.num_channels = int(num_channels)
        self.embeds = nn.ModuleList([nn.Embedding(vocab_size, d_model) for _ in range(self.num_channels)])
        self.channel_combine = channel_combine
        self.channel_proj = nn.Linear(d_model * self.num_channels, d_model) if channel_combine == "concat" else None
        self.emb_drop = nn.Dropout(emb_dropout)

        if patch_depthwise:
            self.patch_proj = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=patch_len, stride=stride, groups=d_model),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            )
        else:
            self.patch_proj = nn.Conv1d(d_model, d_model, kernel_size=patch_len, stride=stride)

        self.pos_embed_patch = nn.Parameter(torch.zeros(1, self.max_patches + 1, d_model))
        nn.init.normal_(self.pos_embed_patch, std=0.02)
        self.pos_drop = nn.Dropout(pos_dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.pre_head_ln = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if head_kind == "linear":
            self.reg_head = nn.Linear(d_model, self.out_dim)
        else:
            self.reg_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(d_model, self.out_dim),
            )

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return _embed_channels(x, tuple(self.embeds), self.channel_combine, self.channel_proj)

    def _encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, dim = tokens.shape
        cls = self.cls_token.expand(bsz, -1, -1)
        p = torch.cat([cls, tokens], dim=1)
        need = num_tokens + 1
        if need <= self.pos_embed_patch.size(1):
            pos = self.pos_embed_patch[:, :need, :]
        else:
            extra = need - self.pos_embed_patch.size(1)
            pad = torch.zeros(1, extra, dim, device=p.device, dtype=p.dtype)
            pos = torch.cat([self.pos_embed_patch, pad], dim=1)
        p = self.pos_drop(p + pos)
        z = self.encoder(p)
        if self.pool == "mean":
            z = z[:, 1:, :].mean(dim=1)
        elif self.pool == "cls":
            z = z[:, 0, :]
        else:
            z = z[:, -1, :]
        z = self.pre_head_ln(z)
        return self.reg_head(z)

    def _forward_from_hidden(self, h: torch.Tensor, repatch: bool = True) -> torch.Tensor:
        assert h.dim() == 3 and h.size(-1) == self.d_model, (
            f"expect hidden input [B, N, {self.d_model}], got {tuple(h.shape)}"
        )
        h = self.emb_drop(h)
        if repatch:
            h = h.transpose(1, 2)
            p = self.patch_proj(h)
            tokens = p.transpose(1, 2)
        else:
            tokens = h
        return self._encode_tokens(tokens)

    def forward_embeddings(self, input_embeds: torch.Tensor, token_mode: Optional[str] = None) -> torch.Tensor:
        use_mode = token_mode or self.token_mode
        if use_mode not in {"repatch", "direct_tokens"}:
            raise ValueError(f"Unsupported token mode {use_mode}")
        return self._forward_from_hidden(input_embeds, repatch=(use_mode == "repatch"))

    def forward(self, x: Optional[torch.Tensor] = None, input_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_embeds is not None:
            return self.forward_embeddings(input_embeds)
        if x is None:
            raise ValueError("PatchTST.forward requires either x or input_embeds.")
        h = self._embed_tokens(x)
        return self._forward_from_hidden(h, repatch=True)

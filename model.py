# -*- coding: utf-8 -*-
# model.py
from typing import Optional, Tuple
import torch
import torch.nn as nn


# Shared helper to embed 4-channel token inputs consistently across models.
def _embed_channels_4(
    x: torch.Tensor,
    embeds: Tuple[nn.Embedding, nn.Embedding, nn.Embedding, nn.Embedding],
    channel_combine: str,
    channel_proj: Optional[nn.Module],
) -> torch.Tensor:
    """Embed [B,T,4] token tensor with four separate embeddings and combine.

    This removes duplicated logic in multiple model classes.
    """
    assert x.dim() == 3 and x.size(-1) == 4, "expect x shape [B,T,4] with channels=[main,ma5,ma10,ma20]"
    embed_main, embed_ma5, embed_ma10, embed_ma20 = embeds
    h_main = embed_main(x[..., 0])
    h5 = embed_ma5(x[..., 1])
    h10 = embed_ma10(x[..., 2])
    h20 = embed_ma20(x[..., 3])
    if channel_combine == "concat":
        assert channel_proj is not None, "channel_proj must be set when using concat"
        h = torch.cat([h_main, h5, h10, h20], dim=-1)
        h = channel_proj(h)
    else:
        h = h_main + h5 + h10 + h20
    return h


class GridTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=60,
        # Bump defaults to a larger-capacity encoder
        d_model=128,
        num_heads=8,
        num_layers=3,
        max_len=300,
        # 合理的可调参数：
        ff_dim=1024,  # 稍小于 2048，综合性能与参数量
        dropout=0.2,  # Transformer 层内 dropout（attn/ffn）
        emb_dropout=0.2,  # 嵌入后（token+pos）dropout
        head_dropout=0.2,  # 回归头内的 dropout
        channel_combine="concat",  # "concat" 或 "sum"
    ):
        super().__init__()
        # 四个通道使用不同的 embedding：主序列、ma5、ma10、ma20
        self.embed_main = nn.Embedding(vocab_size, d_model)
        self.embed_ma5 = nn.Embedding(vocab_size, d_model)
        self.embed_ma10 = nn.Embedding(vocab_size, d_model)
        self.embed_ma20 = nn.Embedding(vocab_size, d_model)
        self.channel_combine = channel_combine
        if channel_combine == "concat":
            self.channel_proj = nn.Linear(d_model * 4, d_model)
        else:
            self.channel_proj = None  # sum 后维度保持 d_model

        self.pos_embed = nn.Embedding(max_len, d_model)  # 学习型位置嵌入
        self.emb_drop = nn.Dropout(emb_dropout)

        # Transformer 编码器；将 dim_feedforward 暴露为可调以便控制容量
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Regression head：输出2维参数 [mu, log_sigma]
        # 说明：为重尾/异方差回归（Student-t NLL）提供均值与尺度参数
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, 2),  # [mu, log_sigma]
        )

    def _embed_channels(self, x):
        """Embed [B,T,4] tokens via four embeddings and combine into [B,T,d_model]."""
        return _embed_channels_4(
            x,
            (self.embed_main, self.embed_ma5, self.embed_ma10, self.embed_ma20),
            self.channel_combine,
            self.channel_proj,
        )

    def forward(self, x):
        # 输入固定为 [B, T, 4]，检查在嵌入函数中完成
        B, T, _ = x.shape
        device = x.device

        h = self._embed_channels(x)  # [B, T, d_model]
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
        h = h + self.pos_embed(pos)  # 注入位置信息
        h = self.emb_drop(h)
        h = self.encoder(h)
        # 对下一步预测，更稳妥的是取最后时间步向量，而非均值
        h_last = h[:, -1, :]
        params = self.reg_head(h_last)  # [B, 2]
        return params


class PatchTST(nn.Module):
    """
    PatchTST-style encoder (closer to common source impl) for sequence-to-one regression
    on tokenized grids.

    Changes vs previous version (Plan A):
    - Pre-norm Transformer (norm_first=True)
    - Learnable patch positional embedding as Parameter + dropout
    - CLS token support and pre-head LayerNorm
    - Keep existing token embedding and Conv1d patching
    """

    def __init__(
        self,
        vocab_size=60,
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
    ):
        super().__init__()
        self.pool = pool
        self.max_patches = int(max_patches)
        self.head_kind = head_kind

        # Token embeddings for 4 channels
        self.embed_main = nn.Embedding(vocab_size, d_model)
        self.embed_ma5 = nn.Embedding(vocab_size, d_model)
        self.embed_ma10 = nn.Embedding(vocab_size, d_model)
        self.embed_ma20 = nn.Embedding(vocab_size, d_model)
        # Combine four channel embeddings; "sum" removes an extra projection layer
        self.channel_combine = channel_combine
        if channel_combine == "concat":
            self.channel_proj = nn.Linear(d_model * 4, d_model)
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
            # 输出 [mu, log_sigma]
            self.reg_head = nn.Linear(d_model, 2)
        else:
            self.reg_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(d_model, 2),  # [mu, log_sigma]
            )

    def _embed_channels(self, x):
        """x: [B,T,4] long -> [B,T,d_model]"""
        return _embed_channels_4(
            x,
            (self.embed_main, self.embed_ma5, self.embed_ma10, self.embed_ma20),
            self.channel_combine,
            self.channel_proj,
        )

    def forward(self, x):
        # 固定 4 通道，形状断言在嵌入函数中完成
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

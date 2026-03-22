from typing import Dict
import torch
import torch.nn as nn

from .ts_vqvae import TSVQVAE


class AttentionPool1D(nn.Module):
    """Attention pooling over token sequences using a learned query."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(x.size(0), -1, -1)
        pooled, _ = self.attn(query, x, x, need_weights=False)
        return pooled


class TSVQVAEFrontTokenizer(nn.Module):
    """Front-end tokenizer for PatchTST ablations.

    Long sequence input: [B, T, F]
    Local embeddings out: [B, N, D]

    Steps per local window:
      1. TS encoder -> z_e [P, D]
      2. optional VQ codebook -> z_q [P, D]
      3. build fusion sequence from z_e / z_q according to ablation config
      4. small fusion encoder -> pooled local embedding
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        local_window: int = 24,
        local_stride: int = 12,
        patch_len: int = 4,
        patch_stride: int = 4,
        codebook_size: int = 256,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        encoder_num_heads: int = 4,
        fusion_num_layers: int = 1,
        fusion_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        recon_weight: float = 1.0,
        commitment_weight: float = 0.25,
        orthogonal_weight: float = 0.0,
        diversity_weight: float = 0.0,
        use_vq: bool = True,
        fusion_mode: str = "ze_zq_fusion",
        zq_pool_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.local_window = int(local_window)
        self.local_stride = int(local_stride)
        self.use_vq = bool(use_vq)
        self.fusion_mode = str(fusion_mode)
        self.zq_pool_mode = str(zq_pool_mode)

        valid_fusion_modes = {"ze_only", "zq_only", "ze_zq_fusion"}
        valid_zq_modes = {"mean", "attn_pool", "full"}
        if self.fusion_mode not in valid_fusion_modes:
            raise ValueError(f"Unsupported fusion_mode={self.fusion_mode}")
        if self.zq_pool_mode not in valid_zq_modes:
            raise ValueError(f"Unsupported zq_pool_mode={self.zq_pool_mode}")
        if not self.use_vq and self.fusion_mode != "ze_only":
            raise ValueError("When use_vq=False, fusion_mode must be 'ze_only' for the cont_only ablation.")

        self.vqvae = TSVQVAE(
            input_dim=self.input_dim,
            window_size=self.local_window,
            patch_len=patch_len,
            patch_stride=patch_stride,
            embed_dim=self.model_dim,
            codebook_size=codebook_size,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            recon_weight=recon_weight,
            commitment_weight=commitment_weight,
            orthogonal_weight=orthogonal_weight,
            diversity_weight=diversity_weight,
        )

        self.zq_attn_pool = AttentionPool1D(self.model_dim, num_heads=fusion_num_heads, dropout=dropout)
        if fusion_num_layers > 0:
            fusion_layer = nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=fusion_num_heads,
                dim_feedforward=int(self.model_dim * mlp_ratio),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=fusion_num_layers)
        else:
            self.fusion = nn.Identity()
        self.output_mlp = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.model_dim, self.model_dim),
        )

    def _split_local_windows(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"expect [B,T,F], got {tuple(x.shape)}"
        if x.size(1) < self.local_window:
            raise ValueError(f"sequence length {x.size(1)} must be >= local_window {self.local_window}")
        windows = x.transpose(1, 2).unfold(-1, self.local_window, self.local_stride)
        windows = windows.permute(0, 2, 3, 1).contiguous()
        return windows

    def _pool_zq(self, z_q: torch.Tensor) -> torch.Tensor:
        if self.zq_pool_mode == "mean":
            return z_q.mean(dim=1, keepdim=True)
        if self.zq_pool_mode == "attn_pool":
            return self.zq_attn_pool(z_q)
        if self.zq_pool_mode == "full":
            return z_q
        raise RuntimeError(f"Unsupported zq_pool_mode={self.zq_pool_mode}")

    def _build_fusion_input(self, z_e: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "ze_only":
            return z_e
        pooled_or_full_zq = self._pool_zq(z_q)
        if self.fusion_mode == "zq_only":
            return pooled_or_full_zq
        if self.fusion_mode == "ze_zq_fusion":
            return torch.cat([pooled_or_full_zq, z_e], dim=1)
        raise RuntimeError(f"Unsupported fusion_mode={self.fusion_mode}")

    def _pool_fusion_output(self, fused: torch.Tensor) -> torch.Tensor:
        h_cls = fused[:, 0, :]
        if fused.size(1) > 1:
            h_mean = fused[:, 1:, :].mean(dim=1)
        else:
            h_mean = h_cls
        return self.output_mlp(torch.cat([h_cls, h_mean], dim=-1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        windows = self._split_local_windows(x)
        bsz, num_tokens, _, _ = windows.shape
        flat_windows = windows.view(bsz * num_tokens, self.local_window, self.input_dim)

        vq_output = self.vqvae(flat_windows, use_vq=self.use_vq)
        z_e = vq_output["z_e"]
        z_q = vq_output["z_q"]
        fusion_in = self._build_fusion_input(z_e, z_q)
        fusion_out = self.fusion(fusion_in)
        local_embeddings = self._pool_fusion_output(fusion_out).view(bsz, num_tokens, self.model_dim)

        embed_ind = vq_output["embed_ind"]
        if embed_ind.dim() == 2:
            embed_ind = embed_ind.view(bsz, num_tokens, -1)

        return {
            "embeddings": local_embeddings,
            "embed_ind": embed_ind,
            "vq_loss": vq_output["vq_loss"],
            "recon_loss": vq_output["recon_loss"],
            "codebook_loss": vq_output["codebook_loss"],
            "commitment_loss": vq_output["commitment_loss"],
            "diversity_loss": vq_output["diversity_loss"],
            "orthogonal_loss": vq_output["orthogonal_loss"],
            "used_codes": vq_output["used_codes"],
            "top1_code_freq": vq_output["top1_code_freq"],
            "z_e": z_e,
            "z_q": z_q,
            "fusion_mode": self.fusion_mode,
            "zq_pool_mode": self.zq_pool_mode,
            "use_vq": torch.tensor(float(self.use_vq), device=x.device),
        }

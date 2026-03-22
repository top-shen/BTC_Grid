from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalTemporalPatchEmbed(nn.Module):
    """Patchify a single-asset multivariate time-series window.

    Input:  [B, W, F]
    Output: [B, P, D]
    """

    def __init__(self, input_dim: int, patch_len: int, patch_stride: int, embed_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.embed_dim = int(embed_dim)
        self.proj = nn.Linear(self.input_dim * self.patch_len, self.embed_dim)

    def num_patches(self, window_size: int) -> int:
        if window_size < self.patch_len:
            raise ValueError(f"window_size={window_size} must be >= patch_len={self.patch_len}")
        return 1 + (window_size - self.patch_len) // self.patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"expect [B,W,F], got {tuple(x.shape)}"
        patches = x.transpose(1, 2).unfold(-1, self.patch_len, self.patch_stride)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(x.size(0), patches.size(1), -1)
        return self.proj(patches)

    def unpatchify(self, patch_values: torch.Tensor, window_size: int) -> torch.Tensor:
        """Average overlapping reconstructed patches back to [B, W, F]."""
        bsz, num_patches, _ = patch_values.shape
        patches = patch_values.view(bsz, num_patches, self.input_dim, self.patch_len)
        patches = patches.permute(0, 1, 3, 2).contiguous()

        recon = patch_values.new_zeros(bsz, window_size, self.input_dim)
        counts = patch_values.new_zeros(bsz, window_size, 1)
        for idx in range(num_patches):
            start = idx * self.patch_stride
            end = start + self.patch_len
            recon[:, start:end, :] += patches[:, idx]
            counts[:, start:end, :] += 1.0
        return recon / counts.clamp_min(1.0)


class VQCodebook(nn.Module):
    """Minimal TS-only vector quantizer adapted from STORM's VQ idea."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
        orthogonal_weight: float = 0.0,
        diversity_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.commitment_weight = float(commitment_weight)
        self.orthogonal_weight = float(orthogonal_weight)
        self.diversity_weight = float(diversity_weight)

        self.codebook = nn.Embedding(self.codebook_size, self.dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def _orthogonal_loss(self) -> torch.Tensor:
        if self.orthogonal_weight <= 0.0:
            return self.codebook.weight.new_zeros(())
        codes = F.normalize(self.codebook.weight, dim=-1)
        sim = torch.matmul(codes, codes.t())
        eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
        off_diag = (sim - eye) ** 2
        return off_diag.sum() / max(1, self.codebook_size * (self.codebook_size - 1))

    def _diversity_loss(self, indices: torch.Tensor) -> torch.Tensor:
        if self.diversity_weight <= 0.0:
            return self.codebook.weight.new_zeros(())
        usage = F.one_hot(indices.reshape(-1), num_classes=self.codebook_size).float().mean(dim=0)
        entropy = -(usage * torch.log(usage.clamp_min(1e-8))).sum()
        max_entropy = torch.log(torch.tensor(float(self.codebook_size), device=usage.device, dtype=usage.dtype))
        return 1.0 - entropy / max_entropy.clamp_min(1e-8)

    def usage_stats(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat = indices.reshape(-1)
        if flat.numel() == 0:
            zero = self.codebook.weight.new_zeros(())
            return {"used_codes": zero, "top1_code_freq": zero}
        counts = torch.bincount(flat, minlength=self.codebook_size).float()
        used_codes = (counts > 0).sum().float()
        top1 = counts.max() / counts.sum().clamp_min(1.0)
        return {"used_codes": used_codes, "top1_code_freq": top1}

    def forward(self, z_e: torch.Tensor):
        assert z_e.dim() == 3, f"expect [B,P,D], got {tuple(z_e.shape)}"
        flat = z_e.reshape(-1, self.dim)
        codebook = self.codebook.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        z_q = self.codebook(indices).view_as(z_e)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = self.commitment_weight * F.mse_loss(z_e, z_q.detach())
        orthogonal_loss = self.orthogonal_weight * self._orthogonal_loss()
        diversity_loss = self.diversity_weight * self._diversity_loss(indices)

        quantized = z_e + (z_q - z_e).detach()
        total_loss = codebook_loss + commitment_loss + orthogonal_loss + diversity_loss
        usage = self.usage_stats(indices)

        return quantized, indices.view(z_e.size(0), z_e.size(1)), total_loss, {
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "orthogonal_loss": orthogonal_loss,
            "diversity_loss": diversity_loss,
            "used_codes": usage["used_codes"],
            "top1_code_freq": usage["top1_code_freq"],
        }


class TSVQVAE(nn.Module):
    """TS-only local-window VQ-VAE used by the front tokenizer.

    Input window: [B, W, F]
    Encoded tokens: z_e [B, P, D]
    Quantized tokens: z_q [B, P, D] if VQ is enabled
    Reconstruction: [B, W, F]
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        patch_len: int,
        patch_stride: int,
        embed_dim: int,
        codebook_size: int,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        recon_weight: float = 1.0,
        commitment_weight: float = 0.25,
        orthogonal_weight: float = 0.0,
        diversity_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.embed_dim = int(embed_dim)
        self.recon_weight = float(recon_weight)

        self.patch_embed = LocalTemporalPatchEmbed(
            input_dim=self.input_dim,
            patch_len=patch_len,
            patch_stride=patch_stride,
            embed_dim=self.embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches(self.window_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.encoder_norm = nn.LayerNorm(self.embed_dim)

        self.quantizer = VQCodebook(
            dim=self.embed_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            orthogonal_weight=orthogonal_weight,
            diversity_weight=diversity_weight,
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_norm = nn.LayerNorm(self.embed_dim)
        self.recon_head = nn.Linear(self.embed_dim, self.patch_embed.input_dim * self.patch_embed.patch_len)

    def encode(self, window: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert window.dim() == 3, f"expect [B,W,F], got {tuple(window.shape)}"
        if window.size(1) != self.window_size:
            raise ValueError(f"expect window size {self.window_size}, got {window.size(1)}")
        tokens = self.patch_embed(window)
        pos = self.pos_embed[:, : tokens.size(1), :]
        z_e = self.encoder_norm(self.encoder(tokens + pos))
        return {"tokens": tokens, "pos": pos, "z_e": z_e}

    def quantize(self, z_e: torch.Tensor, use_vq: bool = True) -> Dict[str, torch.Tensor]:
        if use_vq:
            z_q, embed_ind, _, breakdown = self.quantizer(z_e)
            vq_reg_loss = breakdown["codebook_loss"] + breakdown["commitment_loss"] + breakdown["orthogonal_loss"] + breakdown["diversity_loss"]
            return {
                "z_q": z_q,
                "embed_ind": embed_ind,
                "codebook_loss": breakdown["codebook_loss"],
                "commitment_loss": breakdown["commitment_loss"],
                "orthogonal_loss": breakdown["orthogonal_loss"],
                "diversity_loss": breakdown["diversity_loss"],
                "vq_reg_loss": vq_reg_loss,
                "used_codes": breakdown["used_codes"],
                "top1_code_freq": breakdown["top1_code_freq"],
            }

        dummy_indices = torch.full(z_e.shape[:2], -1, dtype=torch.long, device=z_e.device)
        zero = z_e.new_zeros(())
        return {
            "z_q": z_e,
            "embed_ind": dummy_indices,
            "codebook_loss": zero,
            "commitment_loss": zero,
            "orthogonal_loss": zero,
            "diversity_loss": zero,
            "vq_reg_loss": zero,
            "used_codes": zero,
            "top1_code_freq": zero,
        }

    def decode(self, latent_tokens: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        dec = self.decoder_norm(self.decoder(latent_tokens + pos))
        recon_patches = self.recon_head(dec)
        return self.patch_embed.unpatchify(recon_patches, self.window_size)

    def forward(self, window: torch.Tensor, use_vq: bool = True) -> Dict[str, torch.Tensor]:
        enc = self.encode(window)
        z_e = enc["z_e"]
        quant = self.quantize(z_e, use_vq=use_vq)
        recon = self.decode(quant["z_q"], enc["pos"])
        recon_loss = self.recon_weight * F.mse_loss(recon, window)
        vq_loss = recon_loss + quant["vq_reg_loss"]
        return {
            "z_e": z_e,
            "z_q": quant["z_q"],
            "embed_ind": quant["embed_ind"],
            "recon": recon,
            "recon_loss": recon_loss,
            "codebook_loss": quant["codebook_loss"],
            "commitment_loss": quant["commitment_loss"],
            "diversity_loss": quant["diversity_loss"],
            "orthogonal_loss": quant["orthogonal_loss"],
            "vq_loss": vq_loss,
            "used_codes": quant["used_codes"],
            "top1_code_freq": quant["top1_code_freq"],
        }

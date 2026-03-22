from typing import Dict, Optional
import torch
import torch.nn as nn

from model import PatchTST
from .vq_tokenizer import TSVQVAEFrontTokenizer


class VQTokenizerPatchTST(nn.Module):
    """Compose the VQ tokenizer front-end with the baseline PatchTST predictor."""

    def __init__(self, predictor: PatchTST, tokenizer: TSVQVAEFrontTokenizer) -> None:
        super().__init__()
        self.predictor = predictor
        self.tokenizer = tokenizer

    def forward(self, grid_x: torch.Tensor, ts_x: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if ts_x is None:
            raise ValueError("VQTokenizerPatchTST requires continuous ts_x features.")
        tokenizer_output = self.tokenizer(ts_x)
        pred = self.predictor(input_embeds=tokenizer_output["embeddings"])
        tokenizer_output["pred"] = pred
        return tokenizer_output

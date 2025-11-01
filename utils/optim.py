# -*- coding: utf-8 -*-
import torch


def build_adamw_with_groups(model, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Create AdamW with grouped params: no weight decay for bias/norm/embeddings."""
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = name.lower()
        if n.endswith(".bias") or "norm" in n or "embed" in n or p.ndim < 2:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    return opt


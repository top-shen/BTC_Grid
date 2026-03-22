from typing import Any, Dict

from model import PatchTST
from models.patchtst_adapter import VQTokenizerPatchTST
from models.vq_tokenizer import TSVQVAEFrontTokenizer


def _normalize_input_mode(input_mode: str) -> str:
    input_mode = str(input_mode)
    if input_mode == "vq":
        return "tokenizer"
    return input_mode


def build_model_config_from_args(args) -> Dict[str, Any]:
    return {
        "input_mode": _normalize_input_mode(args.input_mode),
        "use_vq": bool(args.use_vq),
        "fusion_mode": args.fusion_mode,
        "zq_pool_mode": args.zq_pool_mode,
        "patchtst_token_mode": args.patchtst_token_mode,
        "tokenizer_input_mode": args.tokenizer_input_mode,
        "tokenizer": {
            "local_window": args.vq_local_window,
            "local_stride": args.vq_local_stride,
            "patch_len": args.vq_patch_len,
            "patch_stride": args.vq_patch_stride,
            "codebook_size": args.vq_codebook_size,
            "encoder_layers": args.vq_encoder_layers,
            "decoder_layers": args.vq_decoder_layers,
            "encoder_num_heads": args.vq_heads,
            "fusion_num_layers": args.fusion_num_layers,
            "fusion_num_heads": args.fusion_num_heads,
            "mlp_ratio": args.vq_mlp_ratio,
            "dropout": args.vq_dropout,
            "recon_weight": args.vq_recon_weight,
            "commitment_weight": args.vq_commit_weight,
            "orthogonal_weight": args.vq_orthogonal_weight,
            "diversity_weight": args.vq_diversity_weight,
            "use_vq": bool(args.use_vq),
            "fusion_mode": args.fusion_mode,
            "zq_pool_mode": args.zq_pool_mode,
        },
    }


def build_model(num_bins: int, num_channels: int, feature_dim: int, model_config: Dict[str, Any]):
    input_mode = _normalize_input_mode(model_config.get("input_mode", "baseline"))
    token_mode = model_config.get("patchtst_token_mode", "repatch")
    predictor = PatchTST(vocab_size=num_bins, num_channels=num_channels, token_mode=token_mode)

    if input_mode == "baseline":
        return predictor

    tokenizer_cfg = dict(model_config.get("tokenizer", {}))
    model_dim = predictor.d_model
    tokenizer = TSVQVAEFrontTokenizer(input_dim=feature_dim, model_dim=model_dim, **tokenizer_cfg)
    return VQTokenizerPatchTST(predictor=predictor, tokenizer=tokenizer)

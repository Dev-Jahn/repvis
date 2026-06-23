"""Convert V-JEPA 2.1 original checkpoints to HuggingFace format (safetensors).

Usage:
    # Convert all 4 variants
    uv run python convert_vjepa21_to_hf.py --all --output_dir ./hf_models

    # Convert a single variant
    uv run python convert_vjepa21_to_hf.py --model_name vit_base --output_dir ./hf_models/vjepa21-vitb-384
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure project src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from vjepa21_hf.configuration_vjepa21 import VJEPA21Config
from vjepa21_hf.modeling_vjepa21 import VJEPA21Model


# ---------------------------------------------------------------------------
# Model variant configs
# ---------------------------------------------------------------------------

MODEL_VARIANTS = {
    "vit_base": dict(
        checkpoint="checkpoints/vjepa2_1_vitb_dist_vitG_384.pt",
        checkpoint_key="ema_encoder",
        config=dict(
            crop_size=384,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            mlp_ratio=4.0,
            hidden_act="gelu",
            pred_hidden_size=384,
            pred_num_attention_heads=12,
            pred_num_hidden_layers=12,
            pred_num_mask_tokens=8,
            pred_teacher_embed_dim=1664,
            pred_return_all_tokens=True,
            n_output_distillation=1,
        ),
    ),
    "vit_large": dict(
        checkpoint="checkpoints/vjepa2_1_vitl_dist_vitG_384.pt",
        checkpoint_key="ema_encoder",
        config=dict(
            crop_size=384,
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=24,
            mlp_ratio=4.0,
            hidden_act="gelu",
            pred_hidden_size=384,
            pred_num_attention_heads=12,
            pred_num_hidden_layers=12,
            pred_num_mask_tokens=8,
            pred_teacher_embed_dim=1664,
            pred_return_all_tokens=True,
            n_output_distillation=1,
        ),
    ),
    "vit_giant": dict(
        checkpoint="checkpoints/vjepa2_1_vitg_384.pt",
        checkpoint_key="target_encoder",
        config=dict(
            crop_size=384,
            hidden_size=1408,
            num_attention_heads=22,
            num_hidden_layers=40,
            mlp_ratio=48.0 / 11.0,
            hidden_act="gelu",
            pred_hidden_size=384,
            pred_num_attention_heads=12,
            pred_num_hidden_layers=24,
            pred_num_mask_tokens=8,
            pred_return_all_tokens=True,
            n_output_distillation=4,
        ),
    ),
    "vit_gigantic": dict(
        checkpoint="checkpoints/vjepa2_1_vitG_384.pt",
        checkpoint_key="target_encoder",
        config=dict(
            crop_size=384,
            hidden_size=1664,
            num_attention_heads=26,
            num_hidden_layers=48,
            mlp_ratio=64.0 / 13.0,
            hidden_act="gelu",
            pred_hidden_size=384,
            pred_num_attention_heads=12,
            pred_num_hidden_layers=24,
            pred_num_mask_tokens=8,
            pred_return_all_tokens=True,
            n_output_distillation=4,
        ),
    ),
}


# ---------------------------------------------------------------------------
# Key conversion helpers
# ---------------------------------------------------------------------------


def _clean_key(key: str) -> str:
    """Strip module.backbone. prefix from original keys."""
    key = key.replace("module.", "")
    key = key.replace("backbone.", "")
    return key


def convert_encoder_keys(
    og_state_dict: dict[str, torch.Tensor],
    config: VJEPA21Config,
) -> dict[str, torch.Tensor]:
    """Convert original encoder state dict to HF format."""
    hf_sd = {}
    emb_dim = config.hidden_size

    for key, val in og_state_dict.items():
        key = _clean_key(key)

        # Skip positional embedding (we use RoPE)
        if key == "pos_embed":
            continue

        # Patch embeddings
        if key.startswith("patch_embed."):
            hf_key = key.replace("patch_embed.", "encoder.embeddings.patch_embeddings.")
            hf_sd[hf_key] = val
            continue

        # Image patch embeddings
        if key.startswith("patch_embed_img."):
            hf_key = key.replace("patch_embed_img.", "encoder.embeddings.patch_embeddings_img.")
            hf_sd[hf_key] = val
            continue

        # Modality embeddings
        if key == "img_mod_embed":
            hf_sd["encoder.embeddings.img_mod_embed"] = val
            continue
        if key == "video_mod_embed":
            hf_sd["encoder.embeddings.video_mod_embed"] = val
            continue

        # Hierarchical norms
        if key.startswith("norms_block."):
            hf_key = key.replace("norms_block.", "encoder.norms_block.")
            hf_sd[hf_key] = val
            continue

        # Transformer blocks
        if key.startswith("blocks."):
            hf_key = key.replace("blocks.", "encoder.layer.")
            hf_key = hf_key.replace("attn.", "attention.")

            # Split fused QKV
            if "qkv." in hf_key:
                prefix, suffix = hf_key.split("qkv")
                if "bias" in suffix:
                    q, k, v = val[:emb_dim], val[emb_dim : 2 * emb_dim], val[2 * emb_dim :]
                else:
                    q, k, v = val[:emb_dim, :], val[emb_dim : 2 * emb_dim, :], val[2 * emb_dim :, :]
                hf_sd[prefix + "query" + suffix] = q
                hf_sd[prefix + "key" + suffix] = k
                hf_sd[prefix + "value" + suffix] = v
                continue

            # MLP key mapping: SwiGLU uses fc1/fc2/fc3, GELU uses fc1/fc2
            # Original SwiGLU: fc1, fc2 (gate), fc3 (out)
            # Original GELU: fc1, fc2
            # HF GELU: fc1, fc2 (same)
            # HF SwiGLU: fc1, fc2, fc3 (same)
            hf_sd[hf_key] = val
            continue

        # Catch-all (shouldn't normally reach here)
        hf_sd[key] = val

    return hf_sd


def convert_predictor_keys(
    og_state_dict: dict[str, torch.Tensor],
    config: VJEPA21Config,
) -> dict[str, torch.Tensor]:
    """Convert original predictor state dict to HF format."""
    hf_sd = {}
    emb_dim = config.pred_hidden_size

    # Collect mask tokens separately
    mask_tokens = {}

    for key, val in og_state_dict.items():
        key = _clean_key(key)

        # Skip positional embedding
        if key == "predictor_pos_embed":
            continue

        # Predictor embedding
        if key.startswith("predictor_embed."):
            hf_key = key.replace("predictor_embed.", "predictor.embeddings.predictor_embed.")
            hf_sd[hf_key] = val
            continue

        # Mask tokens: collected and stacked
        if key.startswith("mask_tokens."):
            idx = key.split("mask_tokens.")[-1]
            mask_tokens[idx] = val
            continue

        # Modality embeddings
        if key == "img_mod_embed":
            hf_sd["predictor.embeddings.img_mod_embed"] = val
            continue
        if key == "video_mod_embed":
            hf_sd["predictor.embeddings.video_mod_embed"] = val
            continue

        # Predictor norm
        if key.startswith("predictor_norm."):
            hf_key = key.replace("predictor_norm.", "predictor.layernorm.")
            hf_sd[hf_key] = val
            continue

        # Predictor projection
        if key.startswith("predictor_proj."):
            hf_key = key.replace("predictor_proj.", "predictor.proj.")
            hf_sd[hf_key] = val
            continue

        # Context projection
        if key.startswith("predictor_proj_context."):
            hf_key = key.replace("predictor_proj_context.", "predictor.proj_context.")
            hf_sd[hf_key] = val
            continue

        # Predictor blocks
        if key.startswith("predictor_blocks."):
            hf_key = key.replace("predictor_blocks.", "predictor.layer.")
            hf_key = hf_key.replace("attn.", "attention.")

            if "qkv." in hf_key:
                prefix, suffix = hf_key.split("qkv")
                if "bias" in suffix:
                    q, k, v = val[:emb_dim], val[emb_dim : 2 * emb_dim], val[2 * emb_dim :]
                else:
                    q, k, v = val[:emb_dim, :], val[emb_dim : 2 * emb_dim, :], val[2 * emb_dim :, :]
                hf_sd[prefix + "query" + suffix] = q
                hf_sd[prefix + "key" + suffix] = k
                hf_sd[prefix + "value" + suffix] = v
                continue

            hf_sd[hf_key] = val
            continue

    # Stack mask tokens into ParameterList format
    if mask_tokens:
        for idx_str, val in mask_tokens.items():
            hf_sd[f"predictor.embeddings.mask_tokens.{idx_str}"] = val

    return hf_sd


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


@torch.no_grad()
def convert_checkpoint(
    model_name: str,
    output_dir: str,
    base_dir: str = ".",
) -> Path:
    """Convert a single V-JEPA 2.1 checkpoint to HF format."""
    variant = MODEL_VARIANTS[model_name]
    ckpt_path = Path(base_dir) / variant["checkpoint"]

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[{model_name}] Loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Create config
    config = VJEPA21Config(**variant["config"])

    # Extract encoder and predictor state dicts
    encoder_key = variant["checkpoint_key"]
    print(f"[{model_name}] Using encoder key: {encoder_key}")

    encoder_sd_raw = state_dict[encoder_key]
    predictor_sd_raw = state_dict["predictor"]

    # Convert keys
    encoder_sd = convert_encoder_keys(encoder_sd_raw, config)
    predictor_sd = convert_predictor_keys(predictor_sd_raw, config)

    # Merge
    full_sd = {}
    full_sd.update(encoder_sd)
    full_sd.update(predictor_sd)

    # Create model and load
    print(f"[{model_name}] Creating HF model...")
    model = VJEPA21Model(config)
    model_sd = model.state_dict()

    # Debug: check for missing/unexpected keys
    hf_keys = set(model_sd.keys())
    converted_keys = set(full_sd.keys())

    missing = hf_keys - converted_keys
    unexpected = converted_keys - hf_keys

    if missing:
        print(f"[{model_name}] Missing keys ({len(missing)}):")
        for k in sorted(missing):
            print(f"  - {k}")
    if unexpected:
        print(f"[{model_name}] Unexpected keys ({len(unexpected)}):")
        for k in sorted(unexpected):
            print(f"  - {k}")

    # Load with strict=False to handle any minor mismatches, then verify
    info = model.load_state_dict(full_sd, strict=False)
    if info.missing_keys:
        print(f"[{model_name}] Warning - missing after load: {info.missing_keys}")
    if info.unexpected_keys:
        print(f"[{model_name}] Warning - unexpected after load: {info.unexpected_keys}")

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"[{model_name}] Saving to {out_path}")

    # Set auto_map for AutoModel support
    config.auto_map = {
        "AutoConfig": "configuration_vjepa21.VJEPA21Config",
        "AutoModel": "modeling_vjepa21.VJEPA21Model",
    }
    model.save_pretrained(out_path, safe_serialization=True)
    config.save_pretrained(out_path)

    # Copy model code files for trust_remote_code
    import shutil

    src_dir = Path(__file__).resolve().parent / "src" / "vjepa21_hf"
    for fname in ["__init__.py", "configuration_vjepa21.py", "modeling_vjepa21.py"]:
        shutil.copy2(src_dir / fname, out_path / fname)

    print(f"[{model_name}] Done! Saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert V-JEPA 2.1 checkpoints to HF format")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=list(MODEL_VARIANTS.keys()),
        help="Model variant to convert",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all 4 variants",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hf_models",
        help="Output directory for converted models",
    )
    args = parser.parse_args()

    if args.all:
        for name in MODEL_VARIANTS:
            out_dir = str(Path(args.output_dir) / f"vjepa21-{name.replace('vit_', 'vit')}-384")
            try:
                convert_checkpoint(name, out_dir)
            except FileNotFoundError as e:
                print(f"Skipping {name}: {e}")
    elif args.model_name:
        convert_checkpoint(args.model_name, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

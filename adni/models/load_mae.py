"""Load MICCAI 2024 ViT_recipe_for_AD MAE encoder weights into VisionTransformer3D.

Checkpoint layout (official README):
    ckpt["net"] contains MaskedAutoencoderViT3D keys.
Encoder keys that map 1:1 onto our ViT-B:
    patch_embed.proj.*, cls_token, pos_embed, blocks.*, norm.*
Decoder keys (decoder_*, mask_token) and classification head are dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .vit3d import VisionTransformer3D

_SKIP_SUBSTR = (
    "decoder",
    "mask_token",
    "head.",
    "head_dist",
)

_STATE_KEYS = ("net", "model", "state_dict", "encoder")


def extract_state_dict(obj) -> dict:
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint is not a dict: {type(obj)}")
    for key in _STATE_KEYS:
        inner = obj.get(key)
        if isinstance(inner, dict) and inner:
            return inner
    if obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj
    raise KeyError("Could not find a tensor state_dict (tried net/model/state_dict/encoder).")


def _strip_prefix(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "mae.", "encoder.", "vit."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def _keep_encoder_key(name: str) -> bool:
    if any(token in name for token in _SKIP_SUBSTR):
        return False
    return True


def interpolate_pos_embed_3d(pos_embed: torch.Tensor, num_patches: int) -> torch.Tensor:
    """Resize patch-grid pos embed; keep the first (CLS) token."""
    if pos_embed.dim() != 3:
        raise ValueError(f"pos_embed expected [1, N, C], got {tuple(pos_embed.shape)}")
    old_len = int(pos_embed.shape[1])
    dim = int(pos_embed.shape[-1])
    extra = old_len - round((old_len - 1) ** (1.0 / 3.0)) ** 3
    if extra not in (1, 2):
        extra = 1
    old_patches = old_len - extra
    old_grid = round(old_patches ** (1.0 / 3.0))
    new_grid = round(num_patches ** (1.0 / 3.0))
    if old_grid ** 3 != old_patches or new_grid ** 3 != num_patches:
        raise ValueError(
            f"pos_embed grid is not cubic: old_len={old_len} extra={extra} "
            f"num_patches={num_patches}"
        )
    cls_tok = pos_embed[:, :1]
    grid = pos_embed[:, extra : extra + old_patches]
    if old_grid == new_grid:
        return torch.cat([cls_tok, grid], dim=1)
    grid = grid.reshape(1, old_grid, old_grid, old_grid, dim).permute(0, 4, 1, 2, 3)
    grid = F.interpolate(grid, size=(new_grid, new_grid, new_grid), mode="trilinear", align_corners=False)
    grid = grid.permute(0, 2, 3, 4, 1).reshape(1, num_patches, dim)
    return torch.cat([cls_tok, grid], dim=1)


def remap_mae_encoder(raw: dict, model: VisionTransformer3D) -> Tuple[dict, list]:
    """Return tensors that fit `model`, plus skipped original names."""
    target = model.state_dict()
    remapped: Dict[str, torch.Tensor] = {}
    skipped = []
    for raw_name, tensor in raw.items():
        if not torch.is_tensor(tensor):
            continue
        name = _strip_prefix(raw_name)
        if not _keep_encoder_key(name):
            skipped.append(raw_name)
            continue
        if name not in target:
            skipped.append(raw_name)
            continue
        dest = target[name]
        if name == "pos_embed" and tensor.shape != dest.shape:
            tensor = interpolate_pos_embed_3d(tensor, model.patch_embed.num_patches)
        if tuple(tensor.shape) != tuple(dest.shape):
            skipped.append(f"{raw_name} {tuple(tensor.shape)} -> {name} {tuple(dest.shape)}")
            continue
        remapped[name] = tensor
    return remapped, skipped


def load_mae_encoder(model: VisionTransformer3D, ckpt_path: str, verbose: bool = True) -> dict:
    path = Path(ckpt_path)
    if not path.is_file():
        parent = path.parent if path.parent.is_dir() else Path(".")
        nearby = sorted(p.name for p in parent.glob("*") if p.is_file())
        hint = f" files in {parent}: {nearby}" if nearby else f" directory missing or empty: {parent}"
        raise FileNotFoundError(f"MAE checkpoint not found: {path}.{hint}")
    if getattr(model, "dual_stream", False):
        raise RuntimeError("MAE ViT-B is single-stream T1; do not combine with --dual_stream.")
    patch = getattr(model, "patch_embed", None)
    if patch is None or patch.proj.weight.shape[1] != 1:
        raise RuntimeError("MAE ViT-B is 1-channel T1. Train with --mri_only.")

    blob = torch.load(str(path), map_location="cpu")
    raw = extract_state_dict(blob)
    remapped, skipped = remap_mae_encoder(raw, model)
    msg = model.load_state_dict(remapped, strict=False)
    info = {
        "path": str(path),
        "loaded": sorted(remapped.keys()),
        "n_loaded": len(remapped),
        "missing": list(msg.missing_keys),
        "unexpected": list(msg.unexpected_keys),
        "skipped": skipped,
    }
    if verbose:
        print(
            f"MAE encoder loaded from {path}: "
            f"{info['n_loaded']} tensors, "
            f"missing={len(info['missing'])}, skipped={len(skipped)}"
        )
        head_missing = [k for k in info["missing"] if k.startswith("head.")]
        other_missing = [k for k in info["missing"] if not k.startswith("head.")]
        if other_missing:
            print("  missing (non-head):", other_missing[:12])
        if head_missing:
            print("  classification head stays random:", head_missing)
        if info["n_loaded"] < 8:
            raise RuntimeError(
                "Too few MAE tensors matched the ViT. Check --model_size base "
                "and that the file is the official ViT-B MAE checkpoint."
            )
    return info

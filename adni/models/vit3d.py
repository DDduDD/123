"""Lightweight 3D Vision Transformer for multimodal MRI+PET volumes."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn


def _to_3tuple(value: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    value = tuple(int(v) for v in value)
    if len(value) != 3:
        raise ValueError(f"Expected 3 values, got {value}")
    return value


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Sequence[int]] = 96,
        patch_size: Union[int, Sequence[int]] = 16,
        in_chans: int = 2,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = _to_3tuple(img_size)
        self.patch_size = _to_3tuple(patch_size)
        self.grid_size = tuple(s // p for s, p in zip(self.img_size, self.patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,D,H,W] -> [B,N,C]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        qkv_bias: bool = True,
        ln_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=ln_eps)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop, qkv_bias=qkv_bias
        )
        self.norm2 = nn.LayerNorm(dim, eps=ln_eps)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer3D(nn.Module):
    """
    Minimal ViT3D with interfaces expected by Prompt/DOCO:
      - patch_embed
      - _pos_embed
      - blocks / norm / norm_pre
      - head / forward_head
      - embed_dim
    """

    def __init__(
        self,
        img_size: Union[int, Sequence[int]] = 96,
        patch_size: Union[int, Sequence[int]] = 16,
        in_chans: int = 2,
        num_classes: int = 2,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        tabular_dim: int = 0,
        dual_stream: bool = False,
        qkv_bias: bool = True,
        ln_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.tabular_dim = int(tabular_dim)
        self.num_prefix_tokens = 1
        self.dual_stream = bool(dual_stream) and int(in_chans) >= 2
        self.qkv_bias = bool(qkv_bias)
        self.ln_eps = float(ln_eps)

        if self.dual_stream:
            self.patch_embed_mri = PatchEmbed3D(
                img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim
            )
            self.patch_embed_pet = PatchEmbed3D(
                img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim
            )
            n = self.patch_embed_mri.num_patches
            self.num_mri_patches = n
            self.num_pet_patches = n
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 2 * n, embed_dim))
            self.mri_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pet_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.patch_embed = PatchEmbed3D(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            num_patches = self.patch_embed.num_patches
            self.num_mri_patches = num_patches
            self.num_pet_patches = 0
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.mri_type = None
            self.pet_type = None

        self.pos_drop = nn.Dropout(drop_rate)
        self.norm_pre = nn.Identity()

        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    qkv_bias=qkv_bias,
                    ln_eps=ln_eps,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=ln_eps)
        self.head = nn.Linear(embed_dim + self.tabular_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.dual_stream:
            nn.init.trunc_normal_(self.mri_type, std=0.02)
            nn.init.trunc_normal_(self.pet_type, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Patch embed + CLS/pos. Dual-stream keeps MRI then PET tokens."""
        if self.dual_stream:
            if x.size(1) < 2:
                raise ValueError(f"dual_stream expects 2 channels, got {tuple(x.shape)}")
            mri = self.patch_embed_mri(x[:, 0:1])
            pet = self.patch_embed_pet(x[:, 1:2])
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            tokens = torch.cat([cls_token, mri, pet], dim=1)
            tokens = tokens + self.pos_embed
            n = self.num_mri_patches
            tokens[:, 1 : 1 + n] = tokens[:, 1 : 1 + n] + self.mri_type
            tokens[:, 1 + n :] = tokens[:, 1 + n :] + self.pet_type
            return self.pos_drop(tokens)
        patches = self.patch_embed(x)
        return self._pos_embed(patches)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(
        self,
        x: torch.Tensor,
        pre_logits: bool = False,
        tabular: torch.Tensor = None,
    ) -> torch.Tensor:
        feat = x[:, 0]
        if self.tabular_dim > 0:
            if tabular is None:
                tabular = feat.new_zeros(feat.shape[0], self.tabular_dim)
            feat = torch.cat([feat, tabular], dim=-1)
        return feat if pre_logits else self.head(feat)

    def forward(self, x: torch.Tensor, tabular: torch.Tensor = None) -> torch.Tensor:
        x = self.forward_features(x)
        return self.forward_head(x, tabular=tabular)


def create_vit3d(
    model_size: str = "small",
    img_size: int = 96,
    patch_size: int = 16,
    in_chans: int = 2,
    num_classes: int = 2,
    drop_rate: float = 0.0,
    tabular_dim: int = 0,
    dual_stream: bool = False,
    qkv_bias: bool = True,
    ln_eps: float = 1e-6,
) -> VisionTransformer3D:
    configs = {
        "tiny": dict(embed_dim=192, depth=6, num_heads=3),
        "small": dict(embed_dim=384, depth=8, num_heads=6),
        "base": dict(embed_dim=768, depth=12, num_heads=12),
    }
    if model_size not in configs:
        raise ValueError(f"Unknown model_size={model_size}, choose from {list(configs)}")
    cfg = configs[model_size]
    return VisionTransformer3D(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=drop_rate,
        tabular_dim=tabular_dim,
        dual_stream=dual_stream,
        qkv_bias=qkv_bias,
        ln_eps=ln_eps,
        **cfg,
    )

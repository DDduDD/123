"""Prompt wrapper for 3D ViT, adapted from imagenet/method/vpt.py."""

from __future__ import annotations

import math
from functools import reduce
from operator import mul

import torch
import torch.nn as nn

from .vit3d import VisionTransformer3D


class PromptViT3D(nn.Module):
    def __init__(self, vit: VisionTransformer3D, num_prompts: int = 8):
        super().__init__()
        self.vit = vit
        self.num_prompts = num_prompts
        self.prompt_dim = vit.embed_dim
        self.dual_prompt = bool(getattr(vit, "dual_stream", False)) and num_prompts > 0
        self.adapt_pet_prompts = True
        self.prompts = None
        self.prompts_mri = None
        self.prompts_pet = None
        self.num_prompts_mri = 0
        self.num_prompts_pet = 0

        if num_prompts <= 0:
            return
        if self.dual_prompt:
            self.num_prompts_mri = (num_prompts + 1) // 2
            self.num_prompts_pet = num_prompts - self.num_prompts_mri
            self.prompts_mri = nn.Parameter(
                torch.zeros(1, self.num_prompts_mri, self.prompt_dim)
            )
            self.prompts_pet = nn.Parameter(
                torch.zeros(1, self.num_prompts_pet, self.prompt_dim)
            )
            self._init_prompt_(self.prompts_mri)
            self._init_prompt_(self.prompts_pet)
        else:
            self.prompts = nn.Parameter(torch.zeros(1, num_prompts, self.prompt_dim))
            self._init_prompt_(self.prompts)

    def _init_prompt_(self, param: nn.Parameter):
        patch_embed = getattr(self.vit, "patch_embed_mri", None) or self.vit.patch_embed
        patch_numel = reduce(mul, patch_embed.patch_size, 1)
        val = math.sqrt(6.0 / float(3 * patch_numel + self.prompt_dim))
        nn.init.uniform_(param.data, -val, val)

    def reset(self):
        if self.num_prompts <= 0:
            return
        if self.dual_prompt:
            self._init_prompt_(self.prompts_mri)
            self._init_prompt_(self.prompts_pet)
        elif self.prompts is not None:
            self._init_prompt_(self.prompts)

    def prompt_params(self):
        params = []
        if self.prompts is not None:
            params.append(self.prompts)
        if self.prompts_mri is not None:
            params.append(self.prompts_mri)
        if self.prompts_pet is not None and self.adapt_pet_prompts:
            params.append(self.prompts_pet)
        return params

    def freeze_pet_prompts(self):
        self.adapt_pet_prompts = False
        if self.prompts_pet is not None:
            self.prompts_pet.requires_grad_(False)

    def get_prompt_tensor(self) -> torch.Tensor:
        if self.dual_prompt:
            return torch.cat([self.prompts_mri.detach(), self.prompts_pet.detach()], dim=1)
        return self.prompts.detach()

    def set_prompt_tensor(self, tensor: torch.Tensor):
        tensor = tensor.detach()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if self.dual_prompt:
            n = self.num_prompts_mri
            mri = tensor[:, :n].contiguous()
            pet = tensor[:, n:].contiguous()
            self.prompts_mri = nn.Parameter(mri)
            self.prompts_pet = nn.Parameter(pet)
            self.prompts_mri.requires_grad_(True)
            self.prompts_pet.requires_grad_(self.adapt_pet_prompts)
        else:
            self.prompts = nn.Parameter(tensor.contiguous())
            self.prompts.requires_grad_(True)

    def prompt_injection(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_prompts <= 0:
            return x
        if self.dual_prompt:
            n = self.vit.num_mri_patches
            cls_tok = x[:, :1, :]
            mri_tok = x[:, 1 : 1 + n, :]
            pet_tok = x[:, 1 + n :, :]
            p_mri = self.prompts_mri.expand(x.shape[0], -1, -1)
            p_pet = self.prompts_pet.expand(x.shape[0], -1, -1)
            return torch.cat([cls_tok, p_mri, mri_tok, p_pet, pet_tok], dim=1)
        return torch.cat(
            (
                x[:, :1, :],
                self.prompts.expand(x.shape[0], -1, -1),
                x[:, 1:, :],
            ),
            dim=1,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.embed_tokens(x)
        x = self.prompt_injection(x)
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x

    def forward_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.embed_tokens(x)
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x

    def forward(self, x: torch.Tensor, tabular: torch.Tensor = None) -> torch.Tensor:
        x = self.forward_features(x)
        return self.vit.forward_head(x, tabular=tabular)


def unwrap_backbone_state(state_dict: dict) -> dict:
    """Accept raw ViT keys or PromptViT3D keys (`vit.*`)."""
    filtered = {k: v for k, v in state_dict.items() if k != "prompts"}
    if filtered and all(k.startswith("vit.") for k in filtered):
        return {k[4:]: v for k, v in filtered.items()}
    return filtered


def apply_prompts(vit: VisionTransformer3D, num_prompts: int, prompt_tensor=None) -> nn.Module:
    """Wrap a ViT with shallow VPT prompts; optionally copy source-trained tokens."""
    if num_prompts <= 0:
        return vit
    wrapped = PromptViT3D(vit, num_prompts=num_prompts)
    if prompt_tensor is None:
        return wrapped
    wrapped.set_prompt_tensor(prompt_tensor)
    return wrapped

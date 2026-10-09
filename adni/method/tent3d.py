"""TENT (Wang et al., ICLR 2021) for 3D ViT on ADNI.

Original TENT: minimize prediction entropy on each test batch, updating only
normalization affine parameters (BN/GN/LN). ViT3D has LayerNorm, no BatchNorm.

Optional entropy filter (EATA-style): skip high-entropy samples in the backward
pass so unreliable cases do not pollute the update. Inference still runs on
the full batch.
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit


NORM_MODULE_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)


class Tent3D(nn.Module):
    def __init__(self, model, optimizer, steps: int = 1, entropy_filter: bool = False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.entropy_filter = entropy_filter
        self.last_id_ratio = 1.0
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def forward(self, x, tabular=None):
        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(
                x,
                self.model,
                self.optimizer,
                entropy_filter=self.entropy_filter,
                tabular=tabular,
            )
        if self.entropy_filter:
            with torch.no_grad():
                ent = softmax_entropy(outputs.detach())
                thr = math.log(outputs.shape[1]) / 2.0
                self.last_id_ratio = float((ent < thr).float().mean().item())
        else:
            self.last_id_ratio = 1.0
        return outputs

    def reset(self):
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )
        self.last_id_ratio = 1.0


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, entropy_filter: bool = False, tabular=None):
    outputs = model(x, tabular=tabular)
    ent = softmax_entropy(outputs)
    if entropy_filter:
        thr = math.log(outputs.shape[1]) / 2.0
        keep = ent < thr
        if keep.any():
            loss = ent[keep].mean()
        else:
            return outputs.detach()
    else:
        loss = ent.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs.detach()


def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, NORM_MODULE_TYPES):
            for np, p in m.named_parameters(recurse=False):
                if np in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    n_ln = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
            n_ln += 1
    if n_ln == 0:
        raise RuntimeError("TENT found no LayerNorm/GroupNorm/BN to update.")
    return model

"""DOCO for 3D multimodal ADNI, adapted from imagenet/method/doco.py.

Supports a minimal modality-decoupled domain compensation mode:
  - channel0 = MRI, channel1 = PET (early-fusion backbone unchanged)
  - source mean/std cached separately for MRI-only / PET-only views
  - test-time loss_stat = L_stat^MRI + L_stat^PET (+ optional L_reg on full view)
"""

from __future__ import annotations

import math
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from ..data.adni_dataset import encode_tabular
from ..models.prompt_vit3d import PromptViT3D, apply_prompts


def modality_views(x: torch.Tensor):
    """Build MRI-only / PET-only inputs by zeroing the other channel.

    x: [B, 2, D, H, W] with channel0=MRI, channel1=PET.
    """
    if x.dim() != 5 or x.size(1) != 2:
        raise ValueError(f"Expected [B,2,D,H,W], got {tuple(x.shape)}")
    x_mri = x.clone()
    x_mri[:, 1] = 0
    x_pet = x.clone()
    x_pet[:, 0] = 0
    return x_mri, x_pet


def _stat_align_loss(cls_features: torch.Tensor, src_std: torch.Tensor, src_mean: torch.Tensor):
    batch_std, batch_mean = torch.std_mean(cls_features, dim=0)
    loss = torch.norm(batch_std - src_std, p=2) + torch.norm(batch_mean - src_mean, p=2)
    return loss, batch_mean, batch_std


class DOCO3D(nn.Module):
    def __init__(
        self,
        model: PromptViT3D,
        optimizer,
        num_classes: int = 2,
        ema_alpha: float = 0.999,
        warmup_step: int = 50,
        gmm_pool_size: int = 512,
        reg_beta: float = 0.5,
        lr: float = 1e-1,
        closed_set: bool = True,
        modality_decouple: bool = False,
        split_mode: str = "closed",
    ):
        super().__init__()
        self.ema_alpha = ema_alpha
        self.refine_step = 1
        self.warmup_step = warmup_step
        self.lr = lr
        self.reg_beta = reg_beta
        self.num_classes = num_classes
        self.min_samples_for_adapt = 1
        # closed: adapt on full batch (no unknown-class OOD).
        # open: original class-prototype KMeans ID/OOD routing.
        # source_stat: reuse ID-update / OOD-apply, but score by distance to
        #   source CLS mean (domain shift), not unknown-class prototypes.
        if split_mode not in {"closed", "open", "source_stat"}:
            raise ValueError(f"Unknown split_mode={split_mode}")
        # closed_set=False is the old alias for class-prototype open routing.
        if not closed_set and split_mode == "closed":
            split_mode = "open"
        self.split_mode = split_mode
        self.closed_set = split_mode == "closed"
        self.modality_decouple = modality_decouple

        assert hasattr(model, "vit"), "DOCO3D expects PromptViT3D with .vit"
        assert hasattr(model.vit, "head"), "DOCO3D expects classification head"

        self.model = model
        self.optimizer = optimizer
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )
        self.adapted_prompt_stats = None
        self.gmm_pool_size = gmm_pool_size
        self.ood_scores_pool = deque(maxlen=self.gmm_pool_size)
        self.train_info = None
        self.last_id_ratio = 1.0
        self.age_mean = 70.0
        self.age_std = 10.0
        self.use_prompt_for_src_stat = False

    def _has_adapted_prompt(self):
        return self.adapted_prompt_stats is not None

    def _get_saved_prompt_tensor(self):
        return self.adapted_prompt_stats[2].cuda()

    def _load_saved_prompt_into_model(self):
        self.model.set_prompt_tensor(self._get_saved_prompt_tensor())

    def _build_prompt_optimizer(self):
        return torch.optim.AdamW(self.model.prompt_params(), lr=self.lr)

    def _update_prompt_stats(self, batch_mean, batch_std):
        updated_prompt_tensor = self.model.get_prompt_tensor().detach().cpu()
        alpha = self.ema_alpha
        new_values = (batch_mean, batch_std, updated_prompt_tensor)
        for i, new_value in enumerate(new_values):
            self.adapted_prompt_stats[i] = (
                (1 - alpha) * self.adapted_prompt_stats[i] + alpha * new_value
            )

    def _src_std_mean(self, key: str = "both"):
        """Return (std, mean) tensors on CUDA from train_info."""
        info = self.train_info
        if isinstance(info, dict):
            std, mean = info[key]
        else:
            std, mean = info
        return std.cuda(), mean.cuda()

    def _compute_routing_features(self, x):
        raw_cls_features = self.model.forward_raw_features(x)[:, 0]
        # Head may concat Age/Sex; routing uses visual CLS vs visual weight slice.
        prototypes = self.model.vit.head.weight.detach()[:, : self.model.vit.embed_dim]

        if self._has_adapted_prompt():
            self._load_saved_prompt_into_model()
            features_prompted = self.model.forward_features(x)
            routing_cls_features = features_prompted[:, 0]
        else:
            routing_cls_features = raw_cls_features

        cos_sim = F.cosine_similarity(
            routing_cls_features.unsqueeze(1),
            prototypes,
            dim=2,
        )
        max_cos_sim, _ = cos_sim.max(1)
        return raw_cls_features, max_cos_sim

    def _predict_id_mask(self, ood_scores, device):
        current_ood_scores_np = ood_scores.cpu().numpy()
        self.ood_scores_pool.extend(current_ood_scores_np)

        pooled_scores = np.array(self.ood_scores_pool).reshape(-1, 1)
        if pooled_scores.shape[0] < 2:
            return torch.ones(ood_scores.shape[0], dtype=torch.bool, device=device)

        km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(pooled_scores)
        filter_ids = km.predict(current_ood_scores_np.reshape(-1, 1))
        if km.cluster_centers_[0, 0] > km.cluster_centers_[1, 0]:
            filter_ids = 1 - filter_ids
        id_mask = torch.from_numpy(filter_ids == 0).to(device)
        return id_mask

    def _domain_shift_scores(self, raw_cls_features: torch.Tensor) -> torch.Tensor:
        """L2 distance of raw CLS to cached source mean. Larger = more shifted."""
        _, src_mean = self._src_std_mean("both")
        return torch.norm(raw_cls_features - src_mean.unsqueeze(0), p=2, dim=1)

    def _split_batch(self, x):
        with torch.no_grad():
            raw_cls_features, max_cos_sim = self._compute_routing_features(x)
            if self.split_mode == "closed":
                id_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            elif self.split_mode == "source_stat":
                if self.train_info is None:
                    raise RuntimeError("split_mode=source_stat requires obtain_src_stat() first")
                domain_scores = self._domain_shift_scores(raw_cls_features)
                id_mask = self._predict_id_mask(domain_scores, x.device)
            else:
                ood_scores = (1 - max_cos_sim) * 100
                id_mask = self._predict_id_mask(ood_scores, x.device)
            x_id = x[id_mask]
            x_ood = x[~id_mask]
            raw_cls_features_id = raw_cls_features[id_mask]
            self.last_id_ratio = float(id_mask.float().mean().item())
        return {
            "id_mask": id_mask,
            "x_id": x_id,
            "x_ood": x_ood,
            "raw_cls_features_id": raw_cls_features_id,
            "id_ratio": self.last_id_ratio,
        }

    def _compute_raw_id_alignment_loss(self, x_id, raw_cls_features_id):
        """Raw (no-prompt) alignment loss used for logging / bootstrap init stats."""
        if self.modality_decouple:
            with torch.no_grad():
                x_mri, x_pet = modality_views(x_id)
                cls_mri = self.model.forward_raw_features(x_mri)[:, 0]
                cls_pet = self.model.forward_raw_features(x_pet)[:, 0]
            src_std_m, src_mean_m = self._src_std_mean("mri")
            src_std_p, src_mean_p = self._src_std_mean("pet")
            loss_m, _, _ = _stat_align_loss(cls_mri, src_std_m, src_mean_m)
            loss_p, _, _ = _stat_align_loss(cls_pet, src_std_p, src_mean_p)
            loss_raw = loss_m + loss_p
            batch_mean = torch.mean(raw_cls_features_id, dim=0)
            batch_std = torch.std(raw_cls_features_id, dim=0)
            return loss_raw, batch_mean, batch_std

        src_std, src_mean = self._src_std_mean("both")
        return _stat_align_loss(raw_cls_features_id, src_std, src_mean)

    def _refine_existing_prompt(self, x_id, raw_cls_features_id, tabular=None):
        self.model.train()
        self._load_saved_prompt_into_model()
        for param in self.model.prompt_params():
            param.requires_grad_(True)
        self.optimizer = self._build_prompt_optimizer()

        outputs, loss = None, None
        batch_mean_adapted, batch_std_adapted = None, None
        raw_sim_matrix = None
        if x_id.shape[0] > 1 and self.reg_beta > 0:
            with torch.no_grad():
                raw_sim_matrix = pairwise_cosine_matrix(raw_cls_features_id)

        for _ in range(self.refine_step):
            outputs, loss, batch_mean_adapted, batch_std_adapted = forward_and_adapt(
                x_id,
                self.model,
                self.optimizer,
                self.train_info,
                raw_cls_features=raw_cls_features_id,
                reg_beta=self.reg_beta,
                raw_sim_matrix=raw_sim_matrix,
                modality_decouple=self.modality_decouple,
                tabular=tabular,
            )

        self._update_prompt_stats(
            batch_mean_adapted.detach().cpu(),
            batch_std_adapted.detach().cpu(),
        )
        return outputs, loss

    def _bootstrap_new_prompt(self, x_id, raw_cls_features_id, batch_mean, batch_std, tabular=None):
        self.model.train()
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )
        for param in self.model.prompt_params():
            param.requires_grad_(True)
        self.optimizer = self._build_prompt_optimizer()

        outputs, loss = None, None
        raw_sim_matrix = None
        if x_id.shape[0] > 1 and self.reg_beta > 0:
            with torch.no_grad():
                raw_sim_matrix = pairwise_cosine_matrix(raw_cls_features_id)

        for _ in range(self.warmup_step):
            outputs, loss, _, _ = forward_and_adapt(
                x_id,
                self.model,
                self.optimizer,
                self.train_info,
                raw_cls_features=raw_cls_features_id,
                reg_beta=self.reg_beta,
                raw_sim_matrix=raw_sim_matrix,
                modality_decouple=self.modality_decouple,
                tabular=tabular,
            )

        self.adapted_prompt_stats = [
            batch_mean.clone().detach().cpu(),
            batch_std.clone().detach().cpu(),
            self.model.get_prompt_tensor().detach().cpu(),
        ]
        return outputs, loss

    def _infer_id_without_adaptation(self, x_id, tabular=None):
        outputs_id = None
        self.model.eval()
        with torch.no_grad():
            if x_id.shape[0] > 0:
                outputs_id = self.model(x_id, tabular=tabular)
        return outputs_id

    def _infer_ood(self, x_ood, tabular=None):
        outputs_ood = None
        if x_ood.shape[0] > 0:
            self.model.eval()
            with torch.no_grad():
                outputs_ood = self.model(x_ood, tabular=tabular)
        return outputs_ood

    def _merge_outputs(self, outputs_id, outputs_ood, id_mask):
        if outputs_id is None and outputs_ood is None:
            return torch.tensor([]), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        if outputs_id is None:
            return outputs_ood
        if outputs_ood is None:
            return outputs_id

        out_dim = outputs_id.shape[1]
        final_outputs = torch.zeros(id_mask.shape[0], out_dim, device=id_mask.device)
        final_outputs[id_mask] = outputs_id
        final_outputs[~id_mask] = outputs_ood
        return final_outputs

    def forward(self, x, tabular=None):
        split_info = self._split_batch(x)
        x_id = split_info["x_id"]
        x_ood = split_info["x_ood"]
        raw_cls_features_id = split_info["raw_cls_features_id"]
        id_mask = split_info["id_mask"]
        tab_id = tabular[id_mask] if tabular is not None else None
        tab_ood = tabular[~id_mask] if tabular is not None else None

        outputs_id, outputs_ood = None, None
        final_loss_raw = torch.tensor(0.0).cuda()
        final_loss_new = torch.tensor(0.0).cuda()
        final_loss_adapt = torch.tensor(0.0).cuda()

        if x_id.shape[0] > self.min_samples_for_adapt:
            loss_raw, batch_mean, batch_std = self._compute_raw_id_alignment_loss(
                x_id, raw_cls_features_id
            )
            final_loss_raw, final_loss_new = loss_raw, loss_raw
            if self._has_adapted_prompt():
                outputs_id, final_loss_adapt = self._refine_existing_prompt(
                    x_id, raw_cls_features_id, tabular=tab_id
                )
            else:
                outputs_id, final_loss_adapt = self._bootstrap_new_prompt(
                    x_id, raw_cls_features_id, batch_mean, batch_std, tabular=tab_id
                )
        else:
            outputs_id = self._infer_id_without_adaptation(x_id, tabular=tab_id)

        outputs_ood = self._infer_ood(x_ood, tabular=tab_ood)
        final_outputs = self._merge_outputs(outputs_id, outputs_ood, split_info["id_mask"])

        if outputs_id is None and outputs_ood is None:
            return final_outputs
        return (
            final_outputs,
            final_loss_raw,
            final_loss_new,
            final_loss_adapt,
            split_info["id_mask"],
        )

    @torch.no_grad()
    def obtain_src_stat(self, loader: DataLoader, num_samples: int = 300):
        """Estimate source feature mean/std on ADNI train volumes."""
        self.model.eval()
        features_both = []
        features_mri = []
        features_pet = []
        num = 0
        ent_threshold = math.log(self.num_classes) / 2.0 - 1.0

        for batch in loader:
            images = batch[0].cuda()
            tabular = None
            if getattr(self.model.vit, "tabular_dim", 0) > 0 and len(batch) > 2:
                tabular = encode_tabular(batch[2], self.age_mean, self.age_std, images.device)
            feat_fn = (
                self.model.forward_features
                if self.use_prompt_for_src_stat
                else self.model.forward_raw_features
            )
            raw_both = feat_fn(images)
            raw_logits = self.model.vit.forward_head(raw_both, tabular=tabular)
            ent = softmax_entropy(raw_logits)

            if self.num_classes <= 2:
                selected_indices = torch.arange(images.shape[0], device=images.device)
            else:
                selected_indices = torch.where(ent < ent_threshold)[0]
                if selected_indices.numel() == 0:
                    continue

            features_both.append(raw_both[selected_indices, 0].detach().cpu())

            if self.modality_decouple:
                x_mri, x_pet = modality_views(images)
                cls_mri = feat_fn(x_mri)[selected_indices, 0]
                cls_pet = feat_fn(x_pet)[selected_indices, 0]
                features_mri.append(cls_mri.detach().cpu())
                features_pet.append(cls_pet.detach().cpu())

            num += selected_indices.numel()
            if num >= num_samples:
                break

        if not features_both:
            raise RuntimeError("Failed to collect source features for DOCO stats.")

        both = torch.cat(features_both, dim=0)[:num_samples]
        if self.modality_decouple:
            mri = torch.cat(features_mri, dim=0)[:num_samples]
            pet = torch.cat(features_pet, dim=0)[:num_samples]
            self.train_info = {
                "both": torch.std_mean(both, dim=0),
                "mri": torch.std_mean(mri, dim=0),
                "pet": torch.std_mean(pet, dim=0),
            }
            print(
                f"[src_stat] modality_decouple=True "
                f"n={both.shape[0]} dim={both.shape[1]}"
            )
        else:
            self.train_info = torch.std_mean(both, dim=0)
            print(f"[src_stat] joint n={both.shape[0]} dim={both.shape[1]} prompt={self.use_prompt_for_src_stat}")
        return self.train_info

    def reset(self):
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )
        self.adapted_prompt_stats = None
        self.ood_scores_pool.clear()
        self.last_id_ratio = 1.0


def pairwise_cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    temperature = 1
    x = x / temperature
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def configure_model(model, num_prompts: int = 8, source_prompts=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(model, PromptViT3D):
        prompted = model
        if source_prompts is not None:
            prompted = apply_prompts(model.vit, prompted.num_prompts, source_prompts)
    else:
        prompted = apply_prompts(model, num_prompts, source_prompts)
    for param in prompted.parameters():
        param.requires_grad_(False)
    for param in prompted.prompt_params():
        param.requires_grad_(True)
    prompted.to(device)
    prompted.train()
    return prompted


def collect_params(model):
    return model.prompt_params()


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def forward_and_adapt(
    x,
    model: PromptViT3D,
    optimizer,
    train_info,
    raw_cls_features,
    reg_beta,
    raw_sim_matrix=None,
    modality_decouple: bool = False,
    tabular=None,
):
    if modality_decouple:
        if not isinstance(train_info, dict):
            raise RuntimeError("modality_decouple requires dict train_info with mri/pet/both")
        x_mri, x_pet = modality_views(x)
        feat_both = model.forward_features(x)
        feat_mri = model.forward_features(x_mri)
        feat_pet = model.forward_features(x_pet)
        cls_both = feat_both[:, 0]
        cls_mri = feat_mri[:, 0]
        cls_pet = feat_pet[:, 0]

        src_std_m, src_mean_m = train_info["mri"][0].cuda(), train_info["mri"][1].cuda()
        src_std_p, src_mean_p = train_info["pet"][0].cuda(), train_info["pet"][1].cuda()
        loss_mri, _, _ = _stat_align_loss(cls_mri, src_std_m, src_mean_m)
        loss_pet, _, _ = _stat_align_loss(cls_pet, src_std_p, src_mean_p)
        loss = loss_mri + loss_pet

        if reg_beta > 0 and raw_sim_matrix is not None and x.shape[0] > 1:
            prompted_sim = pairwise_cosine_matrix(cls_both)
            loss = loss + reg_beta * F.mse_loss(prompted_sim, raw_sim_matrix)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_std, batch_mean = torch.std_mean(cls_both.detach(), dim=0)
        output = model.vit.forward_head(feat_both.detach(), tabular=tabular)
        return output, loss, batch_mean, batch_std

    # Original joint (early-fusion) DOCO path.
    features_prompted = model.forward_features(x)
    cls_features = features_prompted[:, 0]
    if isinstance(train_info, dict):
        src_std, src_mean = train_info["both"][0].cuda(), train_info["both"][1].cuda()
    else:
        src_std, src_mean = train_info[0].cuda(), train_info[1].cuda()

    loss, batch_mean, batch_std = _stat_align_loss(cls_features, src_std, src_mean)

    if reg_beta > 0 and raw_sim_matrix is not None and x.shape[0] > 1:
        prompted_sim = pairwise_cosine_matrix(cls_features)
        loss = loss + reg_beta * F.mse_loss(prompted_sim, raw_sim_matrix)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    output = model.vit.forward_head(features_prompted.detach(), tabular=tabular)
    return output, loss, batch_mean, batch_std

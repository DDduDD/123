"""DOCO test-time adaptation entry for ADNI multimodal 3D volumes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adni.data.adni_dataset import ADNIMultimodalDataset, encode_tabular
from adni.data.prepare_manifest import build_manifest, save_manifest, summarize
from adni.method import doco3d, tent3d
from adni.models.prompt_vit3d import apply_prompts, unwrap_backbone_state
from adni.models.vit3d import create_vit3d
from torch.utils.data import DataLoader


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Acc / balanced_acc / sens / spec / AUC for binary labels."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    n = max(len(y_true), 1)
    acc = float((y_pred == y_true).mean()) if n else 0.0

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (sens + spec)

    # Vectorized ROC-AUC (ties -> 0.5).
    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        pos = y_score[y_true == 1][:, None]
        neg = y_score[y_true == 0][None, :]
        if pos.size and neg.size:
            gt = (pos > neg).astype(np.float64)
            eq = (pos == neg).astype(np.float64)
            auc = float((gt + 0.5 * eq).mean())

    return {
        "acc": acc,
        "balanced_acc": float(balanced_acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "auc": auc,
        "n": int(n),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def drop_pet_channel(images: torch.Tensor) -> torch.Tensor:
    """Zero PET (channel 1) for missing-PET test. MRI is channel 0."""
    if images.dim() != 5 or images.size(1) < 2:
        raise ValueError(f"--drop_pet expects [B,2,D,H,W], got {tuple(images.shape)}")
    out = images.clone()
    out[:, 1] = 0
    return out


def _tabular_for_model(model, meta, device, age_mean, age_std):
    vit = model
    if hasattr(model, "vit"):
        vit = model.vit
    elif hasattr(model, "model"):
        inner = model.model
        vit = inner.vit if hasattr(inner, "vit") else inner
    if getattr(vit, "tabular_dim", 0) <= 0:
        return None
    return encode_tabular(meta, age_mean, age_std, device)


@torch.no_grad()
def eval_source(model, loader, device, age_mean=70.0, age_std=10.0, drop_pet: bool = False):
    model.eval()
    ys, preds, scores = [], [], []
    for images, labels, meta in loader:
        images = images.to(device)
        if drop_pet:
            images = drop_pet_channel(images)
        labels = labels.to(device)
        tabular = _tabular_for_model(model, meta, device, age_mean, age_std)
        logits = model(images, tabular=tabular)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        ys.append(labels.cpu().numpy())
        preds.append(pred.cpu().numpy())
        scores.append(prob.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_score = np.concatenate(scores)
    return _binary_metrics(y_true, y_pred, y_score)


def eval_online(
    model, loader, device, desc: str = "TTA", age_mean=70.0, age_std=10.0, drop_pet: bool = False
):
    ys, preds, scores = [], [], []
    id_ratios = []
    for images, labels, meta in tqdm(loader, desc=desc):
        images = images.to(device)
        if drop_pet:
            images = drop_pet_channel(images)
        tabular = _tabular_for_model(model, meta, device, age_mean, age_std)
        outputs = model(images, tabular=tabular)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        if logits.numel() == 0:
            continue
        logits = logits.detach()
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        ys.append(labels.cpu().numpy())
        preds.append(pred.cpu().numpy())
        scores.append(prob.detach().cpu().numpy())
        if hasattr(model, "last_id_ratio"):
            id_ratios.append(float(model.last_id_ratio))
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_score = np.concatenate(scores)
    metrics = _binary_metrics(y_true, y_pred, y_score)
    metrics["mean_id_ratio"] = float(np.mean(id_ratios)) if id_ratios else 1.0
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/lz/DOCO-main/data")
    parser.add_argument(
        "--csv",
        type=str,
        default="/home/lz/DOCO-main/data/ADNI.csv",
        help="Path to ADNI.csv (only used if manifest must be rebuilt)",
    )
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument(
        "--split_by",
        type=str,
        default="site",
        choices=["random", "site"],
        help="Used only when manifest needs to be rebuilt",
    )
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/lz/DOCO-main/output_adni/source/best.pt",
    )
    parser.add_argument("--save_dir", type=str, default="/home/lz/DOCO-main/output_adni/doco_tta")
    parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "small", "base"])
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--src_num_samples", type=int, default=300)
    parser.add_argument("--prompt_num", type=int, default=8)
    parser.add_argument("--doco_beta", type=float, default=0.5)
    parser.add_argument("--doco_lr", type=float, default=0.1)
    parser.add_argument("--warmup_step", type=int, default=50)
    parser.add_argument("--ema_alpha", type=float, default=0.999)
    parser.add_argument("--no_zscore", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default="doco",
        choices=["doco", "tent"],
        help="doco: prompt + source-stat alignment. tent: entropy min on LayerNorm (Wang et al., ICLR 2021).",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="closed",
        choices=["closed", "open", "source_stat"],
        help=(
            "DOCO batch split. closed: update on full batch. "
            "open: original class-prototype KMeans (semantic OOD). "
            "source_stat: ID/OOD routing by distance to source CLS mean "
            "(domain-shifted samples get prompt, no gradient)."
        ),
    )
    parser.add_argument(
        "--open_set_routing",
        action="store_true",
        help="Alias for --split_mode open (ImageNet-style class-prototype routing).",
    )
    parser.add_argument(
        "--tent_lr",
        type=float,
        default=1e-3,
        help="TENT SGD learning rate (original cfg uses 1e-3).",
    )
    parser.add_argument(
        "--tent_filter",
        action="store_true",
        help="TENT: skip high-entropy samples in the backward pass (EATA-style).",
    )
    parser.add_argument(
        "--gate_mode",
        type=str,
        default="val",
        choices=["val", "oracle_test", "none"],
        help=(
            "val: gate by checkpoint source val_acc (no test labels). "
            "oracle_test: gate by test source_acc (analysis only). "
            "none: report source/doco only."
        ),
    )
    parser.add_argument(
        "--gate_threshold",
        type=float,
        default=0.70,
        help="If gate signal >= threshold, final result uses source; else DOCO.",
    )
    parser.add_argument(
        "--disable_gate",
        action="store_true",
        help="Deprecated alias for --gate_mode none.",
    )
    parser.add_argument(
        "--modality_decouple",
        action="store_true",
        help=(
            "Line-A minimal multimodal DOCO: cache MRI/PET source mean-std separately "
            "and optimize L_stat^MRI + L_stat^PET (backbone remains early-fusion)."
        ),
    )
    parser.add_argument(
        "--drop_pet",
        action="store_true",
        help=(
            "Test-time missing PET: zero channel 1 on the test loader only. "
            "Source stats still come from dual-modal train volumes."
        ),
    )
    parser.add_argument(
        "--mri_only",
        action="store_true",
        help="Load MRI only. Also inferred from checkpoint in_chans=1.",
    )
    return parser.parse_args()


def apply_gate(signal: float, source_acc: float, doco_acc: float, threshold: float):
    if signal >= threshold:
        return source_acc, "source", signal
    return doco_acc, "doco", signal


def _print_metrics(tag: str, metrics: dict):
    auc = metrics["auc"]
    auc_str = f"{auc:.4f}" if auc == auc else "nan"
    print(
        f"[{tag}] acc={metrics['acc']:.4f} "
        f"bacc={metrics['balanced_acc']:.4f} "
        f"auc={auc_str} "
        f"sens={metrics['sensitivity']:.4f} "
        f"spec={metrics['specificity']:.4f}"
        + (
            f" id_ratio={metrics['mean_id_ratio']:.3f}"
            if "mean_id_ratio" in metrics
            else ""
        )
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.open_set_routing:
        args.split_mode = "open"
    if args.disable_gate:
        args.gate_mode = "none"
    if args.method == "tent":
        args.gate_mode = "none"
    closed_set = args.split_mode == "closed"

    data_root = Path(args.data_root)
    csv_path = Path(args.csv) if args.csv else data_root / "ADNI.csv"
    if args.manifest:
        manifest = Path(args.manifest)
    elif args.split_by == "site":
        manifest = data_root / "manifest_cn_ad_site.csv"
    else:
        manifest = data_root / "manifest_cn_ad_pairs.csv"
    if not manifest.is_file():
        records = build_manifest(
            data_root,
            csv_path=csv_path,
            seed=args.seed,
            split_by=args.split_by,
            test_ratio=args.test_ratio,
        )
        save_manifest(records, manifest)
        print(summarize(records))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model_size = ckpt_args.get("model_size", args.model_size)
    img_size = int(ckpt_args.get("img_size", args.img_size))
    patch_size = int(ckpt_args.get("patch_size", args.patch_size))
    args.img_size = img_size
    args.patch_size = patch_size
    args.model_size = model_size
    ckpt_val_acc = ckpt.get("val_acc", None)
    tabular_dim = int(ckpt.get("tabular_dim", ckpt_args.get("tabular_dim", 0) or 0))
    age_mean = float(ckpt.get("age_mean", 70.0))
    age_std = float(ckpt.get("age_std", 10.0))
    in_chans = int(ckpt.get("in_chans", ckpt_args.get("in_chans", 2) or 2))
    source_prompts = ckpt.get("prompts", None)
    ckpt_prompt_num = int(ckpt.get("prompt_num", ckpt_args.get("prompt_num", 0) or 0))
    if source_prompts is not None:
        source_prompts = source_prompts.float().cpu()
        if source_prompts.dim() == 2:
            source_prompts = source_prompts.unsqueeze(0)
        ckpt_prompt_num = int(source_prompts.shape[1])
    has_source_prompts = source_prompts is not None and ckpt_prompt_num > 0
    dual_stream = bool(ckpt.get("dual_stream", ckpt_args.get("dual_stream", False)))
    if args.mri_only:
        in_chans = 1
    mri_only = in_chans == 1
    if dual_stream and mri_only:
        raise RuntimeError("dual_stream checkpoint cannot be evaluated with MRI-only input.")
    if args.drop_pet and mri_only:
        raise RuntimeError("--drop_pet is for dual-modal checkpoints; MRI-only already has no PET.")
    if args.modality_decouple and mri_only:
        raise RuntimeError("--modality_decouple needs two-channel MRI+PET input.")

    train_set = ADNIMultimodalDataset(
        data_root=data_root,
        manifest_csv=manifest,
        split="train",
        size=(img_size, img_size, img_size),
        use_zscore=not args.no_zscore,
        mri_only=mri_only,
    )
    test_set = ADNIMultimodalDataset(
        data_root=data_root,
        manifest_csv=manifest,
        split="test",
        size=(img_size, img_size, img_size),
        use_zscore=not args.no_zscore,
        mri_only=mri_only,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    base_model = create_vit3d(
        model_size=model_size,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=2,
        tabular_dim=tabular_dim,
        dual_stream=dual_stream,
        qkv_bias=True,
        ln_eps=1e-6,
    )
    base_model.load_state_dict(unwrap_backbone_state(ckpt["model"]))
    if has_source_prompts:
        base_model = apply_prompts(base_model, ckpt_prompt_num, source_prompts)
    base_model.to(device)
    if tabular_dim > 0:
        print(f"tabular Age/Sex enabled dim={tabular_dim} age_mean={age_mean:.2f} age_std={age_std:.2f}")
    if mri_only:
        print("mri_only=True: in_chans=1, PET not loaded")
    print(f"input img_size={img_size} patch_size={patch_size} model_size={model_size}")
    if args.drop_pet:
        print("drop_pet=True: test PET channel zeroed; source stats stay dual-modal")
    token_src = base_model.vit if hasattr(base_model, "num_prompts") else base_model
    if dual_stream:
        print(
            f"dual_stream=True: MRI/PET token sequences "
            f"n={token_src.num_mri_patches}+{token_src.num_pet_patches}"
        )
    if has_source_prompts:
        print(
            f"source prompts loaded n={ckpt_prompt_num} "
            f"shape={tuple(source_prompts.shape)} (eval and TTA use these tokens)"
        )

    source_metrics = eval_source(
        base_model,
        test_loader,
        device,
        age_mean=age_mean,
        age_std=age_std,
        drop_pet=args.drop_pet,
    )
    _print_metrics("source", source_metrics)
    source_acc = source_metrics["acc"]

    if args.method == "tent":
        tent_model = tent3d.configure_model(base_model)
        tent_params, tent_names = tent3d.collect_params(tent_model)
        print(f"TENT updating {len(tent_params)} LN/BN affine params, e.g. {tent_names[:4]}")
        optimizer = torch.optim.SGD(tent_params, lr=args.tent_lr, momentum=0.9)
        adapt_model = tent3d.Tent3D(
            tent_model, optimizer, steps=1, entropy_filter=args.tent_filter
        )
        print(
            f"TENT mode: lr={args.tent_lr} entropy_filter={args.tent_filter} "
            f"(SGD momentum=0.9, match original ImageNet cfg)"
        )
        adapt_metrics = eval_online(
            adapt_model,
            test_loader,
            device,
            desc="TENT",
            age_mean=age_mean,
            age_std=age_std,
            drop_pet=args.drop_pet,
        )
        adapt_tag = "tent"
    else:
        prompted = doco3d.configure_model(
            base_model,
            num_prompts=ckpt_prompt_num if has_source_prompts else args.prompt_num,
        )
        if args.drop_pet and getattr(prompted, "dual_prompt", False):
            prompted.freeze_pet_prompts()
            print("drop_pet: PET prompts frozen; only MRI prompts are updated")
        optimizer = AdamW(doco3d.collect_params(prompted), lr=args.doco_lr)
        adapt_model = doco3d.DOCO3D(
            prompted,
            optimizer,
            num_classes=2,
            ema_alpha=args.ema_alpha,
            warmup_step=args.warmup_step,
            reg_beta=args.doco_beta,
            lr=args.doco_lr,
            closed_set=closed_set,
            modality_decouple=args.modality_decouple,
            split_mode=args.split_mode,
        )
        adapt_model.age_mean = age_mean
        adapt_model.age_std = age_std
        adapt_model.use_prompt_for_src_stat = has_source_prompts
        print(
            f"DOCO mode: split_mode={args.split_mode} "
            f"modality_decouple={args.modality_decouple} "
            f"source_prompts={has_source_prompts} "
            f"dual_prompt={getattr(prompted, 'dual_prompt', False)}"
        )
        print(f"Collecting source stats from train set (n≈{args.src_num_samples}) ...")
        adapt_model.obtain_src_stat(train_loader, num_samples=args.src_num_samples)
        print("Source stats ready.")
        adapt_metrics = eval_online(
            adapt_model,
            test_loader,
            device,
            desc="DOCO-TTA",
            age_mean=age_mean,
            age_std=age_std,
            drop_pet=args.drop_pet,
        )
        adapt_tag = "doco"

    _print_metrics(adapt_tag, adapt_metrics)
    adapt_acc = adapt_metrics["acc"]

    result = {
        "method": args.method,
        "source": source_metrics,
        adapt_tag: adapt_metrics,
        "source_acc": source_acc,
        f"{adapt_tag}_acc": adapt_acc,
        "closed_set": closed_set,
        "split_mode": args.split_mode,
        "modality_decouple": args.modality_decouple,
        "drop_pet": args.drop_pet,
        "mri_only": mri_only,
        "in_chans": in_chans,
        "dual_stream": dual_stream,
        "has_source_prompts": has_source_prompts,
        "prompt_num": ckpt_prompt_num if has_source_prompts else args.prompt_num,
        "checkpoint": str(args.checkpoint),
        "checkpoint_val_acc": ckpt_val_acc,
        "manifest": str(manifest),
        "args": vars(args),
    }

    if args.gate_mode != "none":
        if args.gate_mode == "val":
            if ckpt_val_acc is None:
                raise RuntimeError(
                    "gate_mode=val requires checkpoint['val_acc']. "
                    "Re-train with current train_source.py or use --gate_mode oracle_test."
                )
            signal = float(ckpt_val_acc)
            signal_name = "checkpoint_val_acc"
        else:
            signal = float(source_acc)
            signal_name = "test_source_acc"
        gated_acc, gated_choice, _ = apply_gate(
            signal, source_acc, adapt_acc, args.gate_threshold
        )
        result["gate_mode"] = args.gate_mode
        result["gate_signal_name"] = signal_name
        result["gate_signal"] = signal
        result["gate_threshold"] = args.gate_threshold
        result["gated_choice"] = gated_choice
        result["gated_acc"] = gated_acc
        print(
            f"[gate] mode={args.gate_mode} signal={signal_name}={signal:.4f} "
            f"thr={args.gate_threshold:.2f} choice={gated_choice} gated_acc={gated_acc:.4f}"
        )
    else:
        print("[gate] disabled")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_json = save_dir / "tta_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()

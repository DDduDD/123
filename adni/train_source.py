"""Supervised source training for multimodal 3D ViT on ADNI CN vs AD."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Allow `python adni/train_source.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adni.data.adni_dataset import (
    age_stats_from_samples,
    build_dataloaders,
    class_weights_from_samples,
    encode_tabular,
)
from adni.data.prepare_manifest import build_manifest, save_manifest, summarize
from adni.models.load_mae import load_mae_encoder
from adni.models.prompt_vit3d import PromptViT3D, apply_prompts, unwrap_backbone_state
from adni.models.vit3d import create_vit3d


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device, age_mean: float, age_std: float, use_tabular: bool):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    ys, preds = [], []
    for images, labels, meta in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        tabular = encode_tabular(meta, age_mean, age_std, device) if use_tabular else None
        logits = model(images, tabular=tabular)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        ys.append(labels)
        preds.append(logits.argmax(dim=1))
    y = torch.cat(ys)
    p = torch.cat(preds)
    n = max(int(y.numel()), 1)
    acc = float((p == y).float().mean().item())
    sens = ((p == 1) & (y == 1)).sum().float() / (y == 1).sum().clamp(min=1).float()
    spec = ((p == 0) & (y == 0)).sum().float() / (y == 0).sum().clamp(min=1).float()
    bacc = float((0.5 * (sens + spec)).item())
    return total_loss / n, acc, bacc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/lz/DOCO-main/data")
    parser.add_argument(
        "--csv",
        type=str,
        default="/home/lz/DOCO-main/data/ADNI.csv",
        help="Path to ADNI.csv",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Path to manifest csv. If empty, build it under data_root.",
    )
    parser.add_argument(
        "--split_by",
        type=str,
        default="site",
        choices=["random", "site"],
        help="Used only when manifest needs to be rebuilt",
    )
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default="/home/lz/DOCO-main/output_adni/source")
    parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "small", "base"])
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_zscore", action="store_true")
    parser.add_argument(
        "--no_class_weight",
        action="store_true",
        help="Disable inverse-frequency class weights on the training loss.",
    )
    parser.add_argument(
        "--no_tabular",
        action="store_true",
        help="Disable Age/Sex concatenation on the classification head.",
    )
    parser.add_argument(
        "--mri_only",
        action="store_true",
        help="Train on MRI only (in_chans=1). Not the same as test-time --drop_pet.",
    )
    parser.add_argument(
        "--prompt_num",
        type=int,
        default=8,
        help="Shallow VPT tokens trained with the backbone. 0 disables prompts.",
    )
    parser.add_argument(
        "--dual_stream",
        action="store_true",
        help="Separate MRI/PET patch embeddings and token sequences. Use --prompt_num 0 at source.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="Path to ViT_recipe_for_AD MAE ViT-B encoder (.pth with ckpt['net']). Forces MRI-only.",
    )
    parser.add_argument(
        "--backbone_lr_scale",
        type=float,
        default=0.1,
        help="When --pretrained is set, backbone lr = lr * this; head/prompt keep --lr.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze MAE encoder; train head (and source prompts if --prompt_num > 0).",
    )
    return parser.parse_args()


def build_backbone(args, in_chans: int, tabular_dim: int):
    return create_vit3d(
        model_size=args.model_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=in_chans,
        num_classes=2,
        tabular_dim=tabular_dim,
        dual_stream=args.dual_stream,
        qkv_bias=True,
        ln_eps=1e-6,
    )


def named_trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def build_optimizer(model, args):
    if args.freeze_backbone:
        vit = model.vit if isinstance(model, PromptViT3D) else model
        for name, param in vit.named_parameters():
            if not name.startswith("head."):
                param.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters.")
    if not args.pretrained or args.freeze_backbone:
        return AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    backbone, head = [], []
    for name, param in named_trainable(model):
        if ".head." in name or name.startswith("head."):
            head.append(param)
        elif name.startswith("prompts") or ".prompts" in name:
            head.append(param)
        else:
            backbone.append(param)
    groups = []
    if backbone:
        groups.append(
            {"params": backbone, "lr": args.lr * float(args.backbone_lr_scale)}
        )
    if head:
        groups.append({"params": head, "lr": args.lr})
    return AdamW(groups, weight_decay=args.weight_decay)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tabular = not args.no_tabular
    tabular_dim = 2 if use_tabular else 0
    prompt_num = max(int(args.prompt_num), 0)
    dual_stream = bool(args.dual_stream)
    pretrained = str(args.pretrained or "").strip()
    if pretrained:
        if args.model_size != "base":
            raise RuntimeError("MAE ViT-B weights require --model_size base.")
        if dual_stream:
            raise RuntimeError("MAE ViT-B is single-stream T1; do not use --dual_stream.")
        args.mri_only = True
        print(f"pretrained MAE: {pretrained} -> MRI-only in_chans=1, model_size=base")
        if args.img_size != 128:
            print(
                f"MAE was trained at 128^3; current --img_size={args.img_size} "
                "will interpolate pos_embed."
            )
    in_chans = 1 if args.mri_only else 2
    if dual_stream and args.mri_only:
        raise RuntimeError("--dual_stream needs MRI+PET (do not combine with --mri_only).")
    args.tabular_dim = tabular_dim
    args.in_chans = in_chans
    args.prompt_num = prompt_num
    args.dual_stream = dual_stream
    args.pretrained = pretrained

    data_root = Path(args.data_root)
    csv_path = Path(args.csv) if args.csv else data_root / "ADNI.csv"
    if args.manifest:
        manifest = Path(args.manifest)
    elif args.split_by == "site":
        manifest = data_root / "manifest_cn_ad_site.csv"
    else:
        manifest = data_root / "manifest_cn_ad_pairs.csv"
    if not manifest.is_file():
        print(f"Manifest not found, building: {manifest}")
        records = build_manifest(
            data_root,
            csv_path=csv_path,
            seed=args.seed,
            split_by=args.split_by,
            test_ratio=args.test_ratio,
        )
        save_manifest(records, manifest)
        print(summarize(records))

    loaders = build_dataloaders(
        data_root=data_root,
        manifest_csv=manifest,
        size=(args.img_size, args.img_size, args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_zscore=not args.no_zscore,
        mri_only=args.mri_only,
    )
    train_samples = loaders["train"].dataset.samples
    age_mean, age_std = age_stats_from_samples(train_samples)
    class_weight = class_weights_from_samples(train_samples, num_classes=2).to(device)
    print(
        f"class_weight={class_weight.tolist()} "
        f"age_mean={age_mean:.2f} age_std={age_std:.2f} "
        f"tabular_dim={tabular_dim} in_chans={in_chans} "
        f"prompt_num={prompt_num} dual_stream={dual_stream} "
        f"img_size={args.img_size} pretrained={bool(pretrained)} "
        f"freeze_backbone={args.freeze_backbone}"
    )

    backbone = build_backbone(args, in_chans, tabular_dim)
    if pretrained:
        load_mae_encoder(backbone, pretrained)
    model = apply_prompts(backbone, prompt_num).to(device)

    optimizer = build_optimizer(model, args)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params={n_train} / {n_all}")
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.no_class_weight:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_bacc = -1.0
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_path = save_dir / "best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_acc = 0.0
        running_n = 0

        for images, labels, meta in loaders["train"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            tabular = encode_tabular(meta, age_mean, age_std, device) if use_tabular else None
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, tabular=tabular)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_acc += accuracy_from_logits(logits.detach(), labels) * bs
            running_n += bs

        scheduler.step()
        train_loss = running_loss / max(running_n, 1)
        train_acc = running_acc / max(running_n, 1)
        val_loss, val_acc, val_bacc = evaluate(
            model, loaders["val"], device, age_mean, age_std, use_tabular
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_bacc": val_bacc,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_bacc={val_bacc:.4f} "
            f"time={row['seconds']:.1f}s"
        )

        better = val_bacc > best_val_bacc + 1e-12 or (
            abs(val_bacc - best_val_bacc) <= 1e-12 and val_loss < best_val_loss
        )
        if better:
            best_val_bacc = val_bacc
            best_val_loss = val_loss
            best_val_acc = val_acc
            vit_state = (
                model.vit.state_dict() if isinstance(model, PromptViT3D) else model.state_dict()
            )
            prompt_state = (
                model.get_prompt_tensor().cpu() if isinstance(model, PromptViT3D) else None
            )
            torch.save(
                {
                    "model": vit_state,
                    "prompts": prompt_state,
                    "prompt_num": prompt_num,
                    "args": vars(args),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_bacc": val_bacc,
                    "val_loss": val_loss,
                    "age_mean": age_mean,
                    "age_std": age_std,
                    "tabular_dim": tabular_dim,
                    "in_chans": in_chans,
                    "dual_stream": dual_stream,
                    "pretrained": pretrained,
                    "freeze_backbone": bool(args.freeze_backbone),
                    "class_weight": class_weight.detach().cpu().tolist(),
                },
                best_path,
            )

    backbone = build_backbone(args, in_chans, tabular_dim)
    ckpt = torch.load(best_path, map_location=device)
    backbone.load_state_dict(unwrap_backbone_state(ckpt["model"]))
    test_model = apply_prompts(
        backbone, int(ckpt.get("prompt_num", prompt_num) or 0), ckpt.get("prompts")
    ).to(device)
    test_loss, test_acc, test_bacc = evaluate(
        test_model, loaders["test"], device, age_mean, age_std, use_tabular
    )
    print(
        f"Best val_acc={best_val_acc:.4f} val_bacc={best_val_bacc:.4f} | "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_bacc={test_bacc:.4f}"
    )
    print(f"Saved checkpoint: {best_path}")

    with (save_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "best_val_acc": best_val_acc,
                "best_val_bacc": best_val_bacc,
                "test_acc": test_acc,
                "test_bacc": test_bacc,
                "age_mean": age_mean,
                "age_std": age_std,
                "tabular_dim": tabular_dim,
                "in_chans": in_chans,
                "prompt_num": prompt_num,
                "dual_stream": dual_stream,
                "pretrained": pretrained,
                "freeze_backbone": bool(args.freeze_backbone),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

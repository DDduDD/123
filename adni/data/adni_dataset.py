"""3D transforms and ADNI multimodal dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={data.shape} from {path}")
    return data


def resize_volume(volume: np.ndarray, out_size: Sequence[int]) -> np.ndarray:
    """Trilinear resize using torch (D,H,W) -> out_size."""
    if tuple(volume.shape) == tuple(out_size):
        return volume.astype(np.float32, copy=False)

    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    resized = torch.nn.functional.interpolate(
        tensor,
        size=tuple(int(x) for x in out_size),
        mode="trilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def zscore_nonzero(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mask = volume > 0
    if not np.any(mask):
        return volume
    values = volume[mask]
    mean = float(values.mean())
    std = float(values.std())
    if std < eps:
        out = volume.copy()
        out[mask] = volume[mask] - mean
        return out
    out = volume.copy()
    out[mask] = (volume[mask] - mean) / (std + eps)
    return out


class Compose3D:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, mri: np.ndarray, pet: np.ndarray):
        for transform in self.transforms:
            mri, pet = transform(mri, pet)
        return mri, pet


class ResizePair:
    def __init__(self, size: Sequence[int]):
        self.size = tuple(int(x) for x in size)

    def __call__(self, mri: np.ndarray, pet: np.ndarray):
        return resize_volume(mri, self.size), resize_volume(pet, self.size)


class ZScorePair:
    def __call__(self, mri: np.ndarray, pet: np.ndarray):
        return zscore_nonzero(mri), zscore_nonzero(pet)


class RandomFlip3DPair:
    """Independent random flips along D/H/W, shared by MRI/PET."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, mri: np.ndarray, pet: np.ndarray):
        for axis in (0, 1, 2):
            if np.random.rand() < self.p:
                mri = np.flip(mri, axis=axis).copy()
                pet = np.flip(pet, axis=axis).copy()
        return mri, pet


def build_transforms(split: str, size: Sequence[int], use_zscore: bool = True) -> Compose3D:
    ops: List[Callable] = [ResizePair(size)]
    if use_zscore:
        ops.append(ZScorePair())
    if split == "train":
        ops.append(RandomFlip3DPair(p=0.5))
    return Compose3D(ops)


class ADNIMultimodalDataset(Dataset):
    """
    Returns:
        x: FloatTensor [2, D, H, W]  (channel0=MRI, channel1=PET)
            or [1, D, H, W] when mri_only=True
        y: LongTensor scalar label (CN=0, AD=1)
        meta: dict
    """

    def __init__(
        self,
        data_root: Path,
        manifest_csv: Path,
        split: str,
        size: Sequence[int] = (96, 96, 96),
        use_zscore: bool = True,
        transform: Optional[Compose3D] = None,
        mri_only: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.mri_only = mri_only
        self.size = tuple(int(x) for x in size)
        self.transform = transform or build_transforms(split, self.size, use_zscore=use_zscore)
        self.samples = self._load_split(Path(manifest_csv), split)
        if not self.samples:
            raise RuntimeError(f"No samples for split={split} in {manifest_csv}")

    @staticmethod
    def _load_split(manifest_csv: Path, split: str) -> List[dict]:
        rows: List[dict] = []
        with Path(manifest_csv).open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples[index]
        mri_path = self.data_root / row["mri_path"]
        pet_path = self.data_root / row["pet_path"] if row.get("pet_path") else ""

        mri = load_nifti(mri_path)
        if self.mri_only:
            pet = np.zeros_like(mri)
            mri, pet = self.transform(mri, pet)
            x = mri[None, ...].astype(np.float32)
        else:
            pet = load_nifti(Path(pet_path))
            mri, pet = self.transform(mri, pet)
            x = np.stack([mri, pet], axis=0).astype(np.float32)
        y = int(row["label"])
        meta = {
            "subject_id": row["subject_id"],
            "label_name": row["label_name"],
            "mri_path": str(mri_path),
            "pet_path": str(pet_path),
            "age": row.get("age", ""),
            "sex": row.get("sex", ""),
        }
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), meta


def _as_list(value) -> List:
    if isinstance(value, (list, tuple)):
        return list(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return [value]


def encode_sex(value) -> float:
    text = str(value or "").strip().upper()
    if text in {"M", "MALE", "1"}:
        return 1.0
    if text in {"F", "FEMALE", "0"}:
        return 0.0
    return 0.5


def encode_age(value, mean: float, std: float) -> float:
    try:
        age = float(value)
    except (TypeError, ValueError):
        age = mean
    if std < 1e-6:
        return 0.0
    return (age - mean) / std


def encode_tabular(meta: dict, age_mean: float, age_std: float, device=None) -> torch.Tensor:
    """Batch Age/Sex into [B, 2]: z-scored age, sex (M=1, F=0, missing=0.5)."""
    ages = _as_list(meta.get("age", ""))
    sexes = _as_list(meta.get("sex", ""))
    n = max(len(ages), len(sexes), 1)
    if len(ages) == 1 and n > 1:
        ages = ages * n
    if len(sexes) == 1 and n > 1:
        sexes = sexes * n
    rows = [
        [encode_age(a, age_mean, age_std), encode_sex(s)]
        for a, s in zip(ages, sexes)
    ]
    tensor = torch.tensor(rows, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def age_stats_from_samples(samples: Sequence[dict]) -> Tuple[float, float]:
    ages = []
    for row in samples:
        try:
            ages.append(float(row.get("age") or ""))
        except (TypeError, ValueError):
            continue
    if not ages:
        return 70.0, 10.0
    arr = np.asarray(ages, dtype=np.float64)
    return float(arr.mean()), float(max(arr.std(), 1e-6))


def class_weights_from_samples(samples: Sequence[dict], num_classes: int = 2) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for row in samples:
        counts[int(row["label"])] += 1.0
    total = float(counts.sum().clamp(min=1.0))
    return total / (num_classes * counts.clamp(min=1.0))


def build_dataloaders(
    data_root: Path,
    manifest_csv: Path,
    size: Sequence[int] = (96, 96, 96),
    batch_size: int = 2,
    num_workers: int = 4,
    use_zscore: bool = True,
    mri_only: bool = False,
) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = ADNIMultimodalDataset(
            data_root=data_root,
            manifest_csv=manifest_csv,
            split=split,
            size=size,
            use_zscore=use_zscore,
            mri_only=mri_only,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return loaders

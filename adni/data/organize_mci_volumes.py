"""Flatten nested ADNI MCI nifti exports and optionally z-score normalize."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import nibabel as nib
import numpy as np

from adni.data.adni_dataset import zscore_nonzero
from adni.data.prepare_manifest import normalize_ptid

IMAGE_ID_IN_PATH_RE = re.compile(r"(?:^|[/_\\])I(\d+)(?:[/_\\]|$|\.|_)", re.IGNORECASE)
IMAGE_ID_SUFFIX_RE = re.compile(r"_I(\d+)\.(?:nii(?:\.gz)?|dcm)", re.IGNORECASE)
PTID_IN_PATH_RE = re.compile(r"(?<!\d)(\d{3})_S_(\d{4})(?!\d)")

SourceKind = Literal["nifti", "dicom_dir"]


@dataclass
class SourceRef:
    kind: SourceKind
    path: Path

# folder name -> modality scanned inside that tree
RAW_SUBDIRS: Dict[str, str] = {
    "pMCI_MRI": "MRI",
    "pMCI_PET": "PET",
    "sMCI_MRI": "MRI",
    "sMCI_PET": "PET",
}


def ptid_to_filename(ptid: str) -> str:
    ptid = normalize_ptid(ptid)
    compact = ptid.replace("_", "")
    return f"sub-ADNI{compact}.nii.gz"


def _load_csv_maps(mci_csv_dir: Path) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    specs = [
        ("pMCI_MRI.csv", "MRI", "pMCI"),
        ("pMCI_PET.csv", "PET", "pMCI"),
        ("sMCI_MRI.csv", "MRI", "sMCI"),
        ("sMCI_PET.csv", "PET", "sMCI"),
    ]
    by_image_id: Dict[str, dict] = {}
    by_ptid_modality: Dict[Tuple[str, str], dict] = {}

    for fname, modality, group in specs:
        path = mci_csv_dir / fname
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ptid = normalize_ptid(row.get("Subject ID") or "")
                image_id = (row.get("Image ID") or "").strip()
                if not ptid or not image_id:
                    continue
                meta = {
                    "ptid": ptid,
                    "image_id": image_id,
                    "modality": modality,
                    "group": group,
                    "source_csv": fname,
                }
                if image_id in by_image_id:
                    prev = by_image_id[image_id]
                    raise RuntimeError(
                        f"Duplicate Image ID {image_id}: {prev['source_csv']} vs {fname}"
                    )
                by_image_id[image_id] = meta
                key = (ptid, modality)
                if key in by_ptid_modality:
                    raise RuntimeError(f"Duplicate {modality} row for {ptid} in {fname}")
                by_ptid_modality[key] = meta
    return by_image_id, by_ptid_modality


def extract_image_id_from_path(path: Path) -> Optional[str]:
    text = str(path.as_posix())
    candidates: List[str] = []
    for pattern in (IMAGE_ID_IN_PATH_RE, IMAGE_ID_SUFFIX_RE):
        candidates.extend(pattern.findall(text))
    if not candidates:
        return None
    return max(candidates, key=len)


def extract_ptid_from_path(path: Path) -> Optional[str]:
    match = PTID_IN_PATH_RE.search(str(path.as_posix()))
    if not match:
        return None
    return f"{int(match.group(1)):03d}_S_{int(match.group(2)):04d}"


def _is_dicom(path: Path) -> bool:
    return path.suffix.lower() == ".dcm"


def _dicom_series_dir(dcm_path: Path, image_id: str) -> Path:
    """Prefer the I{image_id} folder that holds the DICOM series."""
    target = f"I{image_id}".upper()
    for parent in [dcm_path.parent, *dcm_path.parents]:
        if parent.name.upper() == target:
            return parent
    return dcm_path.parent


def _iter_dicom_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    if root.is_file() and root.suffix.lower() == ".zip":
        return
    for path in root.rglob("*"):
        if path.is_file() and _is_dicom(path):
            yield path


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _iter_nifti_in_zip(zip_path: Path) -> Iterable[Path]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            lower = name.lower()
            if lower.endswith(".nii") or lower.endswith(".nii.gz"):
                yield zip_path / name


def _iter_nifti_files(root: Path) -> Iterable[Path]:
    seen = set()
    if not root.exists():
        return
    if root.is_file() and root.suffix.lower() == ".zip":
        for path in _iter_nifti_in_zip(root):
            key = str(path)
            if key not in seen:
                seen.add(key)
                yield path
        return
    if root.is_file() and _is_nifti(root):
        yield root
        return
    for path in root.rglob("*"):
        if not path.is_file() or not _is_nifti(path):
            continue
        key = str(path.resolve()) if not str(path).endswith(".zip") else str(path)
        if key in seen:
            continue
        seen.add(key)
        yield path


def open_nifti_any(path: Path) -> nib.Nifti1Image:
    path = Path(path)
    parts = path.parts
    for i, part in enumerate(parts):
        if part.lower().endswith(".zip"):
            zip_path = Path(*parts[: i + 1])
            member = "/".join(parts[i + 1 :]).replace("\\", "/")
            with zipfile.ZipFile(zip_path, "r") as zf:
                data = zf.read(member)
            if member.lower().endswith(".nii.gz"):
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                    raw = gz.read()
                return nib.load(io.BytesIO(raw))
            return nib.load(io.BytesIO(data))
    return nib.load(str(path))


def load_nifti_from_dicom_dir(dicom_dir: Path, dcm2niix_bin: str = "dcm2niix") -> nib.Nifti1Image:
    """Convert one ADNI DICOM series folder to a NIfTI volume in memory."""
    dicom_dir = Path(dicom_dir)
    if shutil.which(dcm2niix_bin) is None:
        raise RuntimeError(
            f"'{dcm2niix_bin}' not found. Install with: conda install -c conda-forge dcm2niix"
        )
    with tempfile.TemporaryDirectory(prefix="mci_dcm_") as tmp:
        tmp_path = Path(tmp)
        cmd = [
            dcm2niix_bin,
            "-z",
            "y",
            "-b",
            "n",
            "-f",
            "converted",
            "-o",
            str(tmp_path),
            str(dicom_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"dcm2niix failed on {dicom_dir}:\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )
        candidates = sorted(
            list(tmp_path.glob("*.nii.gz")) + list(tmp_path.glob("*.nii")),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(f"dcm2niix produced no NIfTI for {dicom_dir}")
        # Read voxel data into memory before temp dir is deleted.
        loaded = nib.load(str(candidates[0]))
        data = np.asanyarray(loaded.dataobj).astype(np.float32)
        return nib.Nifti1Image(data, loaded.affine, loaded.header.copy())


def discover_labeled_roots(raw_root: Path) -> List[Tuple[str, Path, str]]:
    """Return (label, path, modality) for each existing raw tree or zip."""
    raw_root = Path(raw_root)
    found: List[Tuple[str, Path, str]] = []
    for label, modality in RAW_SUBDIRS.items():
        folder = raw_root / label
        zip_path = raw_root / f"{label}.zip"
        if folder.exists():
            found.append((label, folder, modality))
        if zip_path.is_file():
            found.append((f"{label}.zip", zip_path, modality))
    return found


def build_modality_indexes(
    labeled_roots: List[Tuple[str, Path, str]],
) -> Tuple[
    Dict[str, Dict[str, Path]],
    Dict[str, Dict[str, Path]],
    Dict[str, Dict[Tuple[str, str], Path]],
    Dict[str, Dict[Tuple[str, str], Path]],
    dict,
]:
    """
    Returns:
        nifti_by_image_id[modality][image_id]
        dicom_by_image_id[modality][image_id] -> series folder
        nifti_by_ptid[modality][(ptid, modality)]
        dicom_by_ptid[modality][(ptid, modality)] -> series folder
        root_stats
    """
    nifti_by_image_id: Dict[str, Dict[str, Path]] = defaultdict(dict)
    dicom_by_image_id: Dict[str, Dict[str, Path]] = defaultdict(dict)
    nifti_by_ptid: Dict[str, Dict[Tuple[str, str], Path]] = defaultdict(dict)
    dicom_by_ptid: Dict[str, Dict[Tuple[str, str], Path]] = defaultdict(dict)
    root_stats = {}

    for label, root, modality in labeled_roots:
        n_nifti = 0
        n_dicom = 0
        n_nifti_id = 0
        n_dicom_id = 0

        for path in _iter_nifti_files(root):
            n_nifti += 1
            image_id = extract_image_id_from_path(path)
            if image_id and image_id not in nifti_by_image_id[modality]:
                nifti_by_image_id[modality][image_id] = path
                n_nifti_id += 1
            ptid = extract_ptid_from_path(path)
            if ptid:
                key = (ptid, modality)
                if key not in nifti_by_ptid[modality]:
                    nifti_by_ptid[modality][key] = path

        for path in _iter_dicom_files(root):
            n_dicom += 1
            image_id = extract_image_id_from_path(path)
            series_dir = None
            if image_id:
                series_dir = _dicom_series_dir(path, image_id)
                if image_id not in dicom_by_image_id[modality]:
                    dicom_by_image_id[modality][image_id] = series_dir
                    n_dicom_id += 1
            ptid = extract_ptid_from_path(path)
            if ptid and series_dir is not None:
                key = (ptid, modality)
                if key not in dicom_by_ptid[modality]:
                    dicom_by_ptid[modality][key] = series_dir

        root_stats[label] = {
            "modality": modality,
            "path": str(root),
            "nifti_files": n_nifti,
            "dicom_files": n_dicom,
            "nifti_with_image_id": n_nifti_id,
            "dicom_with_image_id": n_dicom_id,
        }

    return nifti_by_image_id, dicom_by_image_id, nifti_by_ptid, dicom_by_ptid, root_stats


def resolve_source(
    meta: dict,
    nifti_by_image_id: Dict[str, Dict[str, Path]],
    dicom_by_image_id: Dict[str, Dict[str, Path]],
    nifti_by_ptid: Dict[str, Dict[Tuple[str, str], Path]],
    dicom_by_ptid: Dict[str, Dict[Tuple[str, str], Path]],
) -> Tuple[Optional[SourceRef], str]:
    modality = meta["modality"]
    image_id = meta["image_id"]
    ptid = meta["ptid"]
    key = (ptid, modality)

    if image_id in nifti_by_image_id.get(modality, {}):
        return SourceRef("nifti", nifti_by_image_id[modality][image_id]), "nifti_image_id"
    if image_id in dicom_by_image_id.get(modality, {}):
        return SourceRef("dicom_dir", dicom_by_image_id[modality][image_id]), "dicom_image_id"
    if key in nifti_by_ptid.get(modality, {}):
        return SourceRef("nifti", nifti_by_ptid[modality][key]), "nifti_ptid"
    if key in dicom_by_ptid.get(modality, {}):
        return SourceRef("dicom_dir", dicom_by_ptid[modality][key]), "dicom_ptid"

    return None, "missing"


def save_volume_from_source(
    src: SourceRef,
    dst_path: Path,
    apply_zscore: bool,
    compress: bool,
    dcm2niix_bin: str,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src.kind == "nifti":
        img = open_nifti_any(src.path)
    else:
        img = load_nifti_from_dicom_dir(src.path, dcm2niix_bin=dcm2niix_bin)

    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {data.shape} from {src.path}")
    if apply_zscore:
        data = zscore_nonzero(data)
    out_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(out_img, str(dst_path if compress else dst_path.with_suffix("")))


def organize_mci_volumes(
    raw_root: Path,
    mci_csv_dir: Path,
    out_root: Path,
    apply_zscore: bool = True,
    compress: bool = True,
    dry_run: bool = False,
    only_modality: str = "ALL",
    dcm2niix_bin: str = "dcm2niix",
) -> dict:
    only_modality = only_modality.upper()
    if only_modality not in {"ALL", "MRI", "PET"}:
        raise ValueError("only_modality must be ALL, MRI, or PET")

    by_image_id, _ = _load_csv_maps(mci_csv_dir)
    labeled_roots = discover_labeled_roots(raw_root)
    if not labeled_roots:
        raise FileNotFoundError(
            f"No pMCI_MRI / pMCI_PET / sMCI_MRI / sMCI_PET folders or zips under {raw_root}"
        )

    nifti_by_image_id, dicom_by_image_id, nifti_by_ptid, dicom_by_ptid, root_stats = (
        build_modality_indexes(labeled_roots)
    )
    stats = {
        "labeled_roots": labeled_roots,
        "root_stats": root_stats,
        "csv_image_ids": len(by_image_id),
        "indexed_mri_nifti": len(nifti_by_image_id.get("MRI", {})),
        "indexed_pet_nifti": len(nifti_by_image_id.get("PET", {})),
        "indexed_pet_dicom": len(dicom_by_image_id.get("PET", {})),
        "written_mri": 0,
        "written_pet": 0,
        "missing": [],
        "skipped_exists": 0,
        "match_by": defaultdict(int),
    }

    for image_id, meta in sorted(by_image_id.items(), key=lambda x: (x[1]["modality"], x[1]["ptid"])):
        if only_modality != "ALL" and meta["modality"] != only_modality:
            continue

        src, how = resolve_source(
            meta, nifti_by_image_id, dicom_by_image_id, nifti_by_ptid, dicom_by_ptid
        )
        if src is None:
            stats["missing"].append(
                {
                    "ptid": meta["ptid"],
                    "modality": meta["modality"],
                    "image_id": image_id,
                    "group": meta["group"],
                    "source_csv": meta["source_csv"],
                }
            )
            continue

        stats["match_by"][how] += 1
        out_name = ptid_to_filename(meta["ptid"])
        dst = out_root / meta["modality"] / out_name

        if dst.is_file() and not dry_run:
            stats["skipped_exists"] += 1
            continue

        if dry_run:
            print(
                f"[dry-run] {meta['group']} {meta['modality']} {meta['ptid']} "
                f"({how}:{src.kind}) <- {src.path}"
            )
        else:
            save_volume_from_source(
                src,
                dst,
                apply_zscore=apply_zscore,
                compress=compress,
                dcm2niix_bin=dcm2niix_bin,
            )
            print(
                f"[ok] {meta['modality']}/{out_name} ({how}:{src.kind}) <- {src.path}"
            )

        if meta["modality"] == "MRI":
            stats["written_mri"] += 1
        else:
            stats["written_pet"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Flatten and normalize ADNI MCI nifti volumes")
    parser.add_argument("--raw_root", type=str, default="/home/lz/DOCO-main/data_MCI")
    parser.add_argument("--mci_csv_dir", type=str, default="/home/lz/DOCO-main/data_MCI/CSV")
    parser.add_argument("--out_root", type=str, default="/home/lz/DOCO-main/data_MCI")
    parser.add_argument(
        "--only_modality",
        type=str,
        default="ALL",
        choices=["ALL", "MRI", "PET"],
        help="Process only MRI or PET rows (useful after MRI already exported)",
    )
    parser.add_argument("--no_zscore", action="store_true")
    parser.add_argument("--no_compress", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--dcm2niix",
        type=str,
        default="dcm2niix",
        help="Path to dcm2niix binary for PET DICOM -> NIfTI conversion",
    )
    args = parser.parse_args()

    stats = organize_mci_volumes(
        raw_root=Path(args.raw_root),
        mci_csv_dir=Path(args.mci_csv_dir),
        out_root=Path(args.out_root),
        apply_zscore=not args.no_zscore,
        compress=not args.no_compress,
        dry_run=args.dry_run,
        only_modality=args.only_modality,
        dcm2niix_bin=args.dcm2niix,
    )

    print("--- raw source scan ---")
    if not stats["root_stats"]:
        print("  <no raw roots found>")
    for label, info in stats["root_stats"].items():
        print(
            f"  {label}: modality={info['modality']} "
            f"nifti={info['nifti_files']} dicom={info['dicom_files']} "
            f"nifti_ids={info['nifti_with_image_id']} dicom_ids={info['dicom_with_image_id']}"
        )

    print("--- summary ---")
    print(f"csv_image_ids: {stats['csv_image_ids']}")
    print(
        f"indexed_mri_nifti: {stats['indexed_mri_nifti']}, "
        f"indexed_pet_nifti: {stats['indexed_pet_nifti']}, "
        f"indexed_pet_dicom: {stats['indexed_pet_dicom']}"
    )
    print(f"written_mri: {stats['written_mri']}, written_pet: {stats['written_pet']}")
    print(f"match_by: {dict(stats['match_by'])}")
    print(f"skipped_exists: {stats['skipped_exists']}")
    print(f"missing: {len(stats['missing'])}")
    if stats["missing"]:
        missing_mri = sum(1 for x in stats["missing"] if x["modality"] == "MRI")
        missing_pet = sum(1 for x in stats["missing"] if x["modality"] == "PET")
        print(f"missing_breakdown: MRI={missing_mri}, PET={missing_pet}")
        for item in stats["missing"][:10]:
            print(f"  missing {item}")
        if len(stats["missing"]) > 10:
            print(f"  ... and {len(stats['missing']) - 10} more")


if __name__ == "__main__":
    main()

"""Build pMCI vs sMCI manifest from ADNI MCI CSV tables.

Expected CSV layout (under --mci_csv_dir):

    pMCI_MRI.csv, pMCI_PET.csv, sMCI_MRI.csv, sMCI_PET.csv

Each file has columns including ``Subject ID``, ``Image ID``, ``Sex``, ``Age``.

Volume files are discovered under ``data_root/MRI`` and ``data_root/PET`` using
the same PTID rules as CN/AD (e.g. ``sub-ADNI002S0729.nii.gz`` for ``002_S_0729``).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from adni.data.prepare_manifest import (
    _assign_split_by_site,
    _assign_split_random,
    _index_modality_dir,
    normalize_ptid,
    save_manifest,
)

MCI_LABEL_MAP = {"pMCI": 1, "sMCI": 0}


def site_from_ptid(ptid: str) -> str:
    """ADNI site is the 3-digit prefix in PTID, e.g. 002_S_0729 -> 002."""
    ptid = normalize_ptid(ptid)
    if not ptid:
        return "UNKNOWN"
    return ptid.split("_", 1)[0]


def _load_modality_csv(path: Path, group: str) -> Dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")

    out: Dict[str, dict] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ptid = normalize_ptid(row.get("Subject ID") or "")
            if not ptid:
                continue
            out[ptid] = {
                "ptid": ptid,
                "group": group,
                "image_id": (row.get("Image ID") or "").strip(),
                "sex": (row.get("Sex") or "").strip(),
                "age": (row.get("Age") or "").strip(),
                "visit": (row.get("Visit") or "").strip(),
                "study_date": (row.get("Study Date") or "").strip(),
                "site": site_from_ptid(ptid),
            }
    return out


def load_mci_metadata(mci_csv_dir: Path) -> Tuple[Dict[str, dict], dict]:
    """Load pMCI/sMCI MRI+PET tables and merge per subject."""
    mci_csv_dir = Path(mci_csv_dir)
    pmci_mri = _load_modality_csv(mci_csv_dir / "pMCI_MRI.csv", "pMCI")
    pmci_pet = _load_modality_csv(mci_csv_dir / "pMCI_PET.csv", "pMCI")
    smci_mri = _load_modality_csv(mci_csv_dir / "sMCI_MRI.csv", "sMCI")
    smci_pet = _load_modality_csv(mci_csv_dir / "sMCI_PET.csv", "sMCI")

    overlap = set(pmci_mri) & set(smci_mri)
    if overlap:
        sample = sorted(overlap)[:5]
        raise RuntimeError(
            f"pMCI and sMCI share {len(overlap)} subjects; check labels. sample={sample}"
        )

    merged: Dict[str, dict] = {}
    stats = {
        "pMCI_mri_rows": len(pmci_mri),
        "pMCI_pet_rows": len(pmci_pet),
        "sMCI_mri_rows": len(smci_mri),
        "sMCI_pet_rows": len(smci_pet),
        "pMCI_mri_pet_intersection": len(set(pmci_mri) & set(pmci_pet)),
        "sMCI_mri_pet_intersection": len(set(smci_mri) & set(smci_pet)),
    }

    for group, mri_map, pet_map in (
        ("pMCI", pmci_mri, pmci_pet),
        ("sMCI", smci_mri, smci_pet),
    ):
        paired = sorted(set(mri_map) & set(pet_map))
        stats[f"{group}_paired_subjects"] = len(paired)
        for ptid in paired:
            mri_row = mri_map[ptid]
            pet_row = pet_map[ptid]
            if mri_row["group"] != pet_row["group"]:
                raise RuntimeError(f"Label conflict for {ptid}: {mri_row['group']} vs {pet_row['group']}")
            merged[ptid] = {
                "ptid": ptid,
                "group": group,
                "label_name": group,
                "label": MCI_LABEL_MAP[group],
                "mri_image_id": mri_row["image_id"],
                "pet_image_id": pet_row["image_id"],
                "sex": mri_row["sex"] or pet_row["sex"],
                "age": mri_row["age"] or pet_row["age"],
                "visit": mri_row["visit"] or pet_row["visit"],
                "study_date": mri_row["study_date"] or pet_row["study_date"],
                "site": mri_row["site"],
            }

    stats["total_paired_subjects"] = len(merged)
    return merged, stats


def build_mci_manifest(
    data_root: Path,
    mci_csv_dir: Path,
    seed: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.2,
    split_by: str = "site",
    debug: bool = False,
) -> List[dict]:
    data_root = Path(data_root)
    mci_csv_dir = Path(mci_csv_dir)
    split_by = (split_by or "site").strip().lower()
    if split_by not in {"random", "site"}:
        raise ValueError(f"Unsupported split_by={split_by}, use random|site")

    meta_by_ptid, csv_stats = load_mci_metadata(mci_csv_dir)
    mri_root = data_root / "MRI"
    pet_root = data_root / "PET"
    for folder, modality in ((mri_root, "MRI"), (pet_root, "PET")):
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing folder: {folder}\n"
                f"Run organize_mci_volumes for {modality} first, e.g.:\n"
                f"  python -m adni.data.organize_mci_volumes "
                f"--raw_root {data_root} --mci_csv_dir {mci_csv_dir} "
                f"--out_root {data_root} --only_modality {modality}"
            )

    mri_map, mri_stats = _index_modality_dir(mri_root, debug=debug)
    pet_map, pet_stats = _index_modality_dir(pet_root, debug=debug)

    paired_ptids = sorted(set(mri_map) & set(pet_map) & set(meta_by_ptid))
    if not paired_ptids:
        csv_only = sorted(meta_by_ptid.keys())[:5]
        mri_only = sorted(set(meta_by_ptid) - set(mri_map))[:5]
        pet_only = sorted(set(meta_by_ptid) - set(pet_map))[:5]
        msg = [
            "No MCI MRI/PET/nii triples found.",
            f"csv_paired={csv_stats['total_paired_subjects']}",
            f"mri_nifti_ptids={len(mri_map)}, pet_nifti_ptids={len(pet_map)}",
            f"mri_nifti_files={mri_stats['nifti_files']}, pet_nifti_files={pet_stats['nifti_files']}",
            f"data_root={data_root}",
            f"mci_csv_dir={mci_csv_dir}",
            f"csv_stats={csv_stats}",
        ]
        if mri_stats["sample_files"]:
            msg.append("MRI sample files: " + ", ".join(mri_stats["sample_files"][:5]))
        if pet_stats["sample_files"]:
            msg.append("PET sample files: " + ", ".join(pet_stats["sample_files"][:5]))
        if csv_only:
            msg.append("CSV sample PTIDs: " + ", ".join(csv_only))
        if mri_only:
            msg.append("Missing MRI nii for sample PTIDs: " + ", ".join(mri_only))
        if pet_only:
            msg.append("Missing PET nii for sample PTIDs: " + ", ".join(pet_only))
        raise RuntimeError("\n".join(msg))

    missing_csv = sorted(set(meta_by_ptid) - set(paired_ptids))
    if missing_csv and debug:
        print(f"[debug] CSV subjects without both nii: {len(missing_csv)}")
        print("  sample:", ", ".join(missing_csv[:10]))

    records: List[dict] = []
    for ptid in paired_ptids:
        meta = meta_by_ptid[ptid]
        records.append(
            {
                "subject_id": ptid,
                "bids_subject": f"sub-ADNI{ptid.replace('_', '')}",
                "label_name": meta["label_name"],
                "label": meta["label"],
                "mri_path": str(mri_map[ptid].relative_to(data_root)).replace("\\", "/"),
                "pet_path": str(pet_map[ptid].relative_to(data_root)).replace("\\", "/"),
                "sex": meta.get("sex", ""),
                "age": meta.get("age", ""),
                "site": meta.get("site", "UNKNOWN"),
                "mmse": "",
                "mri_image_id": meta.get("mri_image_id", ""),
                "pet_image_id": meta.get("pet_image_id", ""),
                "visit": meta.get("visit", ""),
                "study_date": meta.get("study_date", ""),
            }
        )

    holdout_sites = set()
    if split_by == "site":
        records, holdout_sites = _assign_split_by_site(
            records,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    else:
        records = _assign_split_random(
            records,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

    holdout_str = ",".join(sorted(holdout_sites)) if holdout_sites else ""
    for row in records:
        row["split_by"] = split_by
        row["holdout_sites"] = holdout_str
        row["task"] = "pmci_smci"

    records.sort(key=lambda r: (r["split"], r["label_name"], r["subject_id"]))
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare pMCI vs sMCI ADNI manifest")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/lz/DOCO-main/data_MCI",
        help="Root containing MRI/ and PET/ nifti volumes",
    )
    parser.add_argument(
        "--mci_csv_dir",
        type=str,
        default="/home/lz/DOCO-main/data_MCI/CSV",
        help="Directory with pMCI_*.csv and sMCI_*.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output manifest path (default depends on split_by/seed)",
    )
    parser.add_argument("--split_by", type=str, default="site", choices=["random", "site"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    mci_csv_dir = Path(args.mci_csv_dir)
    if args.out:
        out_path = Path(args.out)
    elif args.split_by == "site":
        out_path = data_root / f"manifest_pmci_smci_site_seed{args.seed}.csv"
    else:
        out_path = data_root / "manifest_pmci_smci_pairs.csv"

    records = build_mci_manifest(
        data_root=data_root,
        mci_csv_dir=mci_csv_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_by=args.split_by,
        debug=args.debug,
    )
    save_manifest(records, out_path)

    counts = defaultdict(lambda: defaultdict(int))
    sites = defaultdict(set)
    for row in records:
        counts[row["split"]][row["label_name"]] += 1
        sites[row["split"]].add(row.get("site") or "UNKNOWN")
    lines = [
        f"total_pairs={len(records)}",
        f"task=pmci_smci",
        f"split_by={records[0].get('split_by', 'site') if records else 'site'}",
    ]
    if records and records[0].get("holdout_sites"):
        lines.append(f"holdout_sites={records[0]['holdout_sites']}")
    for split in ("train", "val", "test"):
        pmci = counts[split]["pMCI"]
        smci = counts[split]["sMCI"]
        site_list = ",".join(sorted(sites[split])) if sites[split] else ""
        lines.append(
            f"{split}: pMCI={pmci}, sMCI={smci}, sum={pmci + smci}, "
            f"n_sites={len(sites[split])}, sites={site_list}"
        )
    print("\n".join(lines))
    print(f"mci_csv_dir: {mci_csv_dir}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

"""Build a subject-level MRI+PET paired manifest with train/val/test splits.

Expected layout:

    /home/lz/DOCO-main/data/
      ADNI.csv
      MRI/   # recursive .nii/.nii.gz
      PET/   # recursive .nii/.nii.gz
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SUB_ADNI_RE = re.compile(r"sub-ADNI(\d+)S(\d+)", re.IGNORECASE)
PTID_RE = re.compile(r"(?<!\d)(\d{3})_S_(\d{4})(?!\d)")
COMPACT_RE = re.compile(r"(?<![A-Za-z0-9])(\d{3})S(\d{4})(?![A-Za-z0-9])", re.IGNORECASE)

LABEL_MAP = {"CN": 0, "AD": 1}
ALLOWED_GROUPS = set(LABEL_MAP.keys())


def _strip_nifti_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return name


def normalize_ptid(value: str) -> str:
    """Normalize Subject/PTID/filename to ADNI style `002_S_1261`.

    Accepts:
      - 002_S_1261
      - sub-ADNI002S1261
      - sub-ADNI002S1261.nii.gz
      - 002S1261
    """
    value = _strip_nifti_suffix((value or "").strip())
    if not value:
        return ""
    if re.fullmatch(r"\d{3}_S_\d{4}", value):
        return value

    match = SUB_ADNI_RE.search(value)
    if match:
        return f"{int(match.group(1)):03d}_S_{int(match.group(2)):04d}"

    match = PTID_RE.search(value)
    if match:
        return f"{int(match.group(1)):03d}_S_{int(match.group(2)):04d}"

    match = COMPACT_RE.search(value)
    if match:
        return f"{int(match.group(1)):03d}_S_{int(match.group(2)):04d}"

    return ""


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _iter_nifti_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for path in folder.rglob("*"):
        if path.is_file() and _is_nifti(path):
            files.append(path)
    return sorted(files)


def extract_ptid_from_path(path: Path) -> str:
    """Extract PTID from names like MRI/sub-ADNI002S1261.nii.gz."""
    candidates = [
        path.name,
        _strip_nifti_suffix(path.name),
        path.stem,
    ]
    candidates.extend(path.parts[::-1][:4])
    for text in candidates:
        ptid = normalize_ptid(text)
        if ptid:
            return ptid
    return ""


def _index_modality_dir(folder: Path, debug: bool = False) -> Tuple[Dict[str, Path], dict]:
    """
    Index volumes under a modality root.

    Modality is determined by the root folder (MRI/ or PET/), not by filename.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")

    mapping: Dict[str, Path] = {}
    files = _iter_nifti_files(folder)
    unmatched: List[str] = []

    for path in files:
        ptid = extract_ptid_from_path(path)
        if not ptid:
            unmatched.append(str(path.relative_to(folder)))
            continue
        mapping.setdefault(ptid, path)

    stats = {
        "folder": str(folder),
        "nifti_files": len(files),
        "matched_ptids": len(mapping),
        "unmatched_files": len(unmatched),
        "sample_files": [p.name for p in files[:10]],
        "sample_unmatched": unmatched[:10],
    }
    if debug:
        print(f"[debug] scan {folder}")
        print(f"  nifti_files={stats['nifti_files']} matched_ptids={stats['matched_ptids']} unmatched={stats['unmatched_files']}")
        if stats["sample_files"]:
            print("  sample_files:")
            for name in stats["sample_files"]:
                print(f"    - {name}")
        else:
            print("  sample_files: <none>")
        if stats["sample_unmatched"]:
            print("  sample_unmatched:")
            for name in stats["sample_unmatched"]:
                print(f"    - {name}")
    return mapping, stats


def _load_adni_csv(csv_path: Path) -> Dict[str, dict]:
    """
    Load ADNI.csv and keep one metadata row per PTID.

    Preference order when multiple rows exist:
      1) MRI row
      2) first seen row
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    by_ptid: Dict[str, dict] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = (row.get("Group") or "").strip().upper()
            if group not in ALLOWED_GROUPS:
                continue

            ptid = normalize_ptid(row.get("PTID") or "")
            if not ptid:
                ptid = normalize_ptid(row.get("Subject") or "")
            if not ptid:
                continue

            modality = (row.get("Modality") or "").strip().upper()
            packed = {
                "ptid": ptid,
                "subject": (row.get("Subject") or "").strip(),
                "group": group,
                "sex": (row.get("Sex") or "").strip(),
                "age": (row.get("Age") or "").strip(),
                "site": (row.get("Site") or "").strip(),
                "mmse": (row.get("Closest_MMSE") or "").strip(),
                "modality": modality,
            }

            if ptid not in by_ptid:
                by_ptid[ptid] = packed
            elif by_ptid[ptid].get("modality") != "MRI" and modality == "MRI":
                by_ptid[ptid] = packed
    return by_ptid


def _assign_split_random(
    records: List[dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> List[dict]:
    """Subject-level stratified random split (same distribution)."""
    import random

    by_label: Dict[int, List[dict]] = defaultdict(list)
    for row in records:
        by_label[row["label"]].append(row)

    rng = random.Random(seed)
    for _, rows in by_label.items():
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        n = len(rows)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(n - 1, 1)
        n_val = min(max(n_val, 1), n - n_train - 1) if n - n_train >= 2 else 0
        for rank, i in enumerate(idx):
            if rank < n_train:
                split = "train"
            elif rank < n_train + n_val:
                split = "val"
            else:
                split = "test"
            rows[i]["split"] = split
    return [r for rows in by_label.values() for r in rows]


def _pick_holdout_sites(
    records: List[dict],
    seed: int,
    test_ratio: float,
    min_test_subjects: int = 20,
) -> set:
    """
    Choose holdout sites whose subject count is about test_ratio of all subjects.
    Prefer sites that keep both CN and AD present in train and test when possible.
    """
    import random

    by_site: Dict[str, List[dict]] = defaultdict(list)
    for row in records:
        site = (row.get("site") or "").strip() or "UNKNOWN"
        row["site"] = site
        by_site[site].append(row)

    sites = list(by_site.keys())
    if len(sites) < 2:
        raise RuntimeError(
            f"Need at least 2 sites for site split, got {len(sites)}: {sites}"
        )

    rng = random.Random(seed)
    rng.shuffle(sites)

    total = len(records)
    target = max(min_test_subjects, int(round(total * test_ratio)))
    target = min(target, total - min_test_subjects)

    holdout = []
    holdout_n = 0
    for site in sites:
        # Keep at least one site for train/val.
        if len(holdout) >= len(sites) - 1:
            break
        holdout.append(site)
        holdout_n += len(by_site[site])
        if holdout_n >= target:
            break

    if holdout_n == 0 or holdout_n >= total:
        raise RuntimeError("Failed to construct a valid site holdout split.")

    # Soft check: warn via return; caller prints summary.
    return set(holdout)


def _assign_split_by_site(
    records: List[dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[dict], set]:
    """
    Domain split:
      - hold out some sites entirely as test (domain shift)
      - remaining sites: subject-level stratified train/val
    """
    import random

    holdout_sites = _pick_holdout_sites(records, seed=seed, test_ratio=test_ratio)

    test_rows = []
    remain_rows = []
    for row in records:
        site = (row.get("site") or "").strip() or "UNKNOWN"
        row["site"] = site
        if site in holdout_sites:
            row["split"] = "test"
            test_rows.append(row)
        else:
            remain_rows.append(row)

    # Within source sites, split train/val only (no extra test).
    # Reuse ratios renormalized over remain set: train:(train+val), val rest.
    by_label: Dict[int, List[dict]] = defaultdict(list)
    for row in remain_rows:
        by_label[row["label"]].append(row)

    rng = random.Random(seed + 7)
    train_vs_val = train_ratio / max(train_ratio + val_ratio, 1e-6)

    for _, rows in by_label.items():
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        n = len(rows)
        if n == 1:
            rows[0]["split"] = "train"
            continue
        n_train = int(round(n * train_vs_val))
        n_train = min(max(n_train, 1), n - 1)
        for rank, i in enumerate(idx):
            rows[i]["split"] = "train" if rank < n_train else "val"

    out = test_rows + [r for rows in by_label.values() for r in rows]
    return out, holdout_sites


def build_manifest(
    data_root: Path,
    csv_path: Optional[Path] = None,
    seed: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.2,
    split_by: str = "random",
    debug: bool = False,
) -> List[dict]:
    """
    Pair MRI/PET by PTID using:
      - labels/metadata from ADNI.csv
      - volume files under data_root/MRI and data_root/PET

    split_by:
      - random: subject stratified iid split
      - site: hold out sites as test domain
    """
    data_root = Path(data_root)
    csv_path = Path(csv_path) if csv_path else data_root / "ADNI.csv"
    mri_root = data_root / "MRI"
    pet_root = data_root / "PET"
    split_by = (split_by or "random").strip().lower()
    if split_by not in {"random", "site"}:
        raise ValueError(f"Unsupported split_by={split_by}, use random|site")

    meta_by_ptid = _load_adni_csv(csv_path)
    mri_map, mri_stats = _index_modality_dir(mri_root, debug=debug)
    pet_map, pet_stats = _index_modality_dir(pet_root, debug=debug)

    paired_ptids = sorted(set(mri_map) & set(pet_map) & set(meta_by_ptid))
    if not paired_ptids:
        msg = [
            "No MRI/PET/CSV triples found.",
            f"mri_matched={len(mri_map)}, pet_matched={len(pet_map)}, csv={len(meta_by_ptid)}",
            f"mri_nifti_files={mri_stats['nifti_files']}, pet_nifti_files={pet_stats['nifti_files']}",
            f"csv_path={csv_path}",
            f"mri_root={mri_root}",
            f"pet_root={pet_root}",
        ]
        if mri_stats["sample_files"]:
            msg.append("MRI sample files: " + ", ".join(mri_stats["sample_files"][:5]))
        else:
            msg.append("MRI sample files: <none>  (目录为空，或没有 .nii/.nii.gz)")
        if pet_stats["sample_files"]:
            msg.append("PET sample files: " + ", ".join(pet_stats["sample_files"][:5]))
        else:
            msg.append("PET sample files: <none>  (目录为空，或没有 .nii/.nii.gz)")
        if mri_stats["sample_unmatched"]:
            msg.append("MRI unmatched samples: " + ", ".join(mri_stats["sample_unmatched"][:5]))
        if pet_stats["sample_unmatched"]:
            msg.append("PET unmatched samples: " + ", ".join(pet_stats["sample_unmatched"][:5]))
        raise RuntimeError("\n".join(msg))

    records: List[dict] = []
    for ptid in paired_ptids:
        meta = meta_by_ptid[ptid]
        group = meta["group"]
        records.append(
            {
                "subject_id": ptid,
                "bids_subject": meta.get("subject", ""),
                "label_name": group,
                "label": LABEL_MAP[group],
                "mri_path": str(mri_map[ptid].relative_to(data_root)).replace("\\", "/"),
                "pet_path": str(pet_map[ptid].relative_to(data_root)).replace("\\", "/"),
                "sex": meta.get("sex", ""),
                "age": meta.get("age", ""),
                "site": (meta.get("site") or "").strip() or "UNKNOWN",
                "mmse": meta.get("mmse", ""),
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

    # Attach split metadata for summarize callers via attribute on list is awkward;
    # store on each row instead.
    holdout_str = ",".join(sorted(holdout_sites)) if holdout_sites else ""
    for row in records:
        row["split_by"] = split_by
        row["holdout_sites"] = holdout_str

    records.sort(key=lambda r: (r["split"], r["label_name"], r["subject_id"]))
    return records


def save_manifest(records: List[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "bids_subject",
        "label_name",
        "label",
        "split",
        "split_by",
        "holdout_sites",
        "mri_path",
        "pet_path",
        "sex",
        "age",
        "site",
        "mmse",
        # Optional MCI task metadata (empty for CN/AD manifests)
        "task",
        "mri_image_id",
        "pet_image_id",
        "visit",
        "study_date",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def summarize(records: List[dict]) -> str:
    counts = defaultdict(lambda: defaultdict(int))
    sites = defaultdict(set)
    for row in records:
        counts[row["split"]][row["label_name"]] += 1
        sites[row["split"]].add(row.get("site") or "UNKNOWN")

    split_by = records[0].get("split_by", "random") if records else "random"
    holdout = records[0].get("holdout_sites", "") if records else ""
    lines = [
        f"total_pairs={len(records)}",
        f"split_by={split_by}",
    ]
    if holdout:
        lines.append(f"holdout_sites={holdout}")
    for split in ("train", "val", "test"):
        cn = counts[split]["CN"]
        ad = counts[split]["AD"]
        site_list = ",".join(sorted(sites[split])) if sites[split] else ""
        lines.append(
            f"{split}: CN={cn}, AD={ad}, sum={cn + ad}, n_sites={len(sites[split])}, sites={site_list}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Prepare ADNI MRI+PET manifest")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/lz/DOCO-main/data",
        help="Dataset root containing ADNI.csv, MRI/, PET/",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Path to ADNI.csv (default: <data_root>/ADNI.csv)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output csv path (default depends on split_by)",
    )
    parser.add_argument(
        "--split_by",
        type=str,
        default="random",
        choices=["random", "site"],
        help="random=iid subject split; site=hold out sites as test domain",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Approximate subject ratio for holdout sites when split_by=site",
    )
    parser.add_argument("--debug", action="store_true", help="Print scan details")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    csv_path = Path(args.csv) if args.csv else data_root / "ADNI.csv"
    if args.out:
        out_path = Path(args.out)
    elif args.split_by == "site":
        out_path = data_root / "manifest_cn_ad_site.csv"
    else:
        out_path = data_root / "manifest_cn_ad_pairs.csv"

    records = build_manifest(
        data_root,
        csv_path=csv_path,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_by=args.split_by,
        debug=args.debug,
    )
    save_manifest(records, out_path)
    print(summarize(records))
    print(f"csv: {csv_path}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

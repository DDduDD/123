"""Download MICCAI 2024 ViT_recipe_for_AD MAE ViT-B (75% mask, BRATS+IXI+OASIS3).

Official Google Drive:
  https://drive.google.com/file/d/1vSxBZ78NXdcAklyFtJPOHtP2ttpUMTQ8

Kaggle copy (same checkpoint name):
  https://www.kaggle.com/datasets/gunasekharnitap/vit-b-pretrained-noaug-mae75-brats2023-ixi-oasis3
"""

from __future__ import annotations

import argparse
from pathlib import Path

FILE_ID = "1vSxBZ78NXdcAklyFtJPOHtP2ttpUMTQ8"
DEFAULT_NAME = "vitb_mae75_brats_ixi_oasis3.pth"
DRIVE_URL = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"
KAGGLE_URL = "https://www.kaggle.com/datasets/gunasekharnitap/vit-b-pretrained-noaug-mae75-brats2023-ixi-oasis3"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="/home/lz/DOCO-main/pretrained/" + DEFAULT_NAME,
    )
    return parser.parse_args()


def _offline_help(out: Path) -> str:
    return (
        "This machine cannot reach drive.google.com (Errno 101).\n"
        "Download on a computer that can open Google, then copy the file here.\n"
        f"  browser: {DRIVE_URL}\n"
        f"  kaggle:  {KAGGLE_URL}\n"
        "  expected size ~500MB, official key is ckpt['net']\n"
        f"  scp the file to: {out}\n"
        "If this cluster has a proxy:\n"
        "  export https_proxy=http://USER:PASS@HOST:PORT\n"
        f"  gdown {FILE_ID} -O {out}"
    )


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size > 10_000_000:
        print(f"Already exists: {out} ({out.stat().st_size} bytes)")
        return
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "Need gdown: pip install gdown\n"
            f"Then: gdown {FILE_ID} -O {out}\n\n"
            + _offline_help(out)
        ) from exc
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print(f"Downloading {url} -> {out}")
    try:
        gdown.download(url, str(out), quiet=False, fuzzy=True)
    except Exception as exc:
        raise SystemExit(_offline_help(out) + f"\n\nOriginal error: {exc}") from exc
    if not out.is_file() or out.stat().st_size < 10_000_000:
        raise SystemExit("Download finished but file is missing or too small.\n" + _offline_help(out))
    print(f"Saved {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

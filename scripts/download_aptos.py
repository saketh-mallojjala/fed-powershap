"""Download APTOS 2019 from Kaggle and lay it out under ``data_cache/aptos2019/``.

Prereqs:
  1. Accept the competition rules (one-time, manual):
       https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules
  2. Place a Kaggle API token at ``~/.kaggle/kaggle.json``
       (Kaggle → Account → Create New API Token)
  3. ``pip install kaggle``

Run:
    python scripts/download_aptos.py

Layout produced:
    data_cache/aptos2019/
        train.csv
        train_images/<id_code>.png

Test CSV/images from the Kaggle bundle are ignored — Kaggle's test labels are
private, so we re-split ``train.csv`` ourselves (see ``data/aptos.py``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile

COMPETITION = "aptos2019-blindness-detection"
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")


def main(root: str = DEFAULT_ROOT) -> None:
    target = os.path.join(root, "aptos2019")
    os.makedirs(target, exist_ok=True)

    csv_path = os.path.join(target, "train.csv")
    img_dir = os.path.join(target, "train_images")
    if os.path.exists(csv_path) and os.path.isdir(img_dir) and os.listdir(img_dir):
        print(f"[skip] APTOS already present at {target}")
        return

    cred_json = os.path.expanduser("~/.kaggle/kaggle.json")
    cred_token = os.path.expanduser("~/.kaggle/access_token")
    if not (os.path.exists(cred_json) or os.path.exists(cred_token) or os.environ.get("KAGGLE_API_TOKEN")):
        sys.exit(
            f"[error] No Kaggle credentials found.\n"
            f"  Either {cred_json} (legacy) or {cred_token} / $KAGGLE_API_TOKEN (new) is required.\n"
            "  Create one at https://www.kaggle.com/settings (Create New API Token)."
        )
    # The kaggle CLI reads $KAGGLE_API_TOKEN if present; pull it from the file.
    if not os.environ.get("KAGGLE_API_TOKEN") and os.path.exists(cred_token):
        with open(cred_token) as f:
            os.environ["KAGGLE_API_TOKEN"] = f.read().strip()

    # Resolve the kaggle binary that lives next to the running python (same venv).
    bin_dir = os.path.dirname(sys.executable)
    kaggle_bin = os.path.join(bin_dir, "kaggle")
    if not os.path.exists(kaggle_bin):
        kaggle_bin = shutil.which("kaggle") or "kaggle"

    print(f"[kaggle] downloading {COMPETITION} into {target} ...")
    cmd = [kaggle_bin, "competitions", "download", "-c", COMPETITION, "-p", target]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("[error] `kaggle` CLI not found. Run: pip install kaggle")
    except subprocess.CalledProcessError as e:
        sys.exit(
            f"[error] kaggle download failed (exit={e.returncode}). "
            "Did you accept the competition rules on the website?"
        )

    zip_path = os.path.join(target, f"{COMPETITION}.zip")
    if not os.path.exists(zip_path):
        sys.exit(f"[error] expected {zip_path} after download")

    print(f"[unzip] {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)

    # Clean up the test split (we re-split train.csv ourselves) + the zip.
    for junk in ("test.csv", "sample_submission.csv", "test_images"):
        p = os.path.join(target, junk)
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
    os.remove(zip_path)

    n = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
    print(f"[done] {csv_path}, {n} images in {img_dir}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    main(root)

"""Pre-decode + resize all APTOS PNGs into a single uint8 tensor on disk.

Without this cache, every batch in every FL round re-opens and re-decodes
each PNG from disk — that's the main bottleneck on Mac (3-5x slower than the
actual model compute on MPS). The cache eliminates that cost: training reads
slices from one in-memory tensor instead.

Run once:
    python scripts/prep_aptos_cache.py                 # 224x224 (default)
    python scripts/prep_aptos_cache.py --image-size 192

Output: data_cache/aptos2019/cache_<image_size>.pt
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def main(data_root: str, image_size: int, force: bool = False) -> None:
    base = os.path.join(data_root, "aptos2019")
    csv = os.path.join(base, "train.csv")
    img_dir = os.path.join(base, "train_images")
    out = os.path.join(base, f"cache_{image_size}.pt")

    if not (os.path.exists(csv) and os.path.isdir(img_dir)):
        sys.exit(
            f"[error] APTOS not found at {base}. "
            f"Run scripts/download_aptos.py first."
        )
    if os.path.exists(out) and not force:
        print(f"[skip] cache already exists: {out} (pass --force to rebuild)")
        return

    df = pd.read_csv(csv)
    n = len(df)
    print(f"[prep] {n} images -> {image_size}x{image_size} uint8 cache at {out}")

    resize = transforms.Resize((image_size, image_size))

    images = torch.empty((n, 3, image_size, image_size), dtype=torch.uint8)
    for i, row in tqdm(df.iterrows(), total=n, desc="decoding"):
        path = os.path.join(img_dir, f"{row['id_code']}.png")
        img = Image.open(path).convert("RGB")
        img = resize(img)
        arr = np.asarray(img, dtype=np.uint8)
        images[i] = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    labels = torch.tensor(df["diagnosis"].astype(int).values, dtype=torch.long)
    id_codes = df["id_code"].astype(str).tolist()

    torch.save(
        {
            "images": images,
            "labels": labels,
            "id_codes": id_codes,
            "image_size": image_size,
        },
        out,
    )
    size_gb = os.path.getsize(out) / 1e9
    print(f"[done] {out} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data_cache")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    main(args.data_root, args.image_size, args.force)

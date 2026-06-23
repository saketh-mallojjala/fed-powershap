"""Turn raw BraTS MRI volumes into a 2D slice cache for HGG/LGG classification.

BraTS is a 3D segmentation dataset; we reframe it as binary tumor-grade
classification (HGG vs LGG) on a handful of tumor-bearing axial slices per
subject. This script decodes the NIfTI volumes once and writes a uint8 tensor
cache that ``data/brats.py`` reads at train time.

Expected raw layout (BraTS 2018/2019/2020 "training" folder)::

    <src>/
        name_mapping.csv                  (Grade, BraTS_2020_subject_ID, ...)
        BraTS20_Training_001/
            BraTS20_Training_001_flair.nii.gz
            BraTS20_Training_001_t1ce.nii.gz
            BraTS20_Training_001_seg.nii.gz
            ...
        BraTS20_Training_002/
            ...

Grade (HGG/LGG) comes from ``name_mapping.csv``. For each subject we pick the
``--slices-per-subject`` axial slices with the largest tumor area (from the seg
mask), take the chosen modality, min-max normalize to uint8, resize, and
replicate to 3 channels (so the ImageNet-pretrained ResNet input is unchanged).

Run once::

    python scripts/prep_brats_slices.py --src /path/to/BraTS2020/training
    python scripts/prep_brats_slices.py --src ... --image-size 224 --modality flair

Output: data_cache/brats/cache_<image_size>.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch
from tqdm import tqdm


def _load_grade_map(src: str) -> dict:
    """Map subject_id -> 1 (HGG) / 0 (LGG) from name_mapping.csv."""
    import pandas as pd

    candidates = glob.glob(os.path.join(src, "**", "name_mapping*.csv"), recursive=True)
    if not candidates:
        sys.exit(
            f"[error] no name_mapping*.csv under {src}. "
            "It carries the HGG/LGG grade labels and is required."
        )
    df = pd.read_csv(candidates[0])
    grade_col = next((c for c in df.columns if c.lower() == "grade"), None)
    id_col = next((c for c in df.columns if "subject_id" in c.lower()), None)
    if grade_col is None or id_col is None:
        sys.exit(f"[error] could not find Grade / subject_id columns in {candidates[0]}")
    mapping = {}
    for _, row in df.iterrows():
        mapping[str(row[id_col])] = 1 if str(row[grade_col]).upper() == "HGG" else 0
    return mapping


def _find_volume(subject_dir: str, suffix: str):
    hits = glob.glob(os.path.join(subject_dir, f"*{suffix}.nii*"))
    return hits[0] if hits else None


def _to_uint8(slice2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice2d, [1, 99])
    if hi <= lo:
        hi = slice2d.max() if slice2d.max() > lo else lo + 1
    norm = np.clip((slice2d - lo) / (hi - lo), 0, 1)
    return (norm * 255).astype(np.uint8)


def main(args) -> None:
    try:
        import nibabel as nib
    except ImportError:
        sys.exit("[error] nibabel required: pip install nibabel")

    from torchvision.transforms import v2

    out_base = os.path.join(args.data_root, "brats")
    os.makedirs(out_base, exist_ok=True)
    out = os.path.join(out_base, f"cache_{args.image_size}.pt")
    if os.path.exists(out) and not args.force:
        print(f"[skip] cache already exists: {out} (pass --force to rebuild)")
        return

    grade_map = _load_grade_map(args.src)
    subject_dirs = sorted(
        d for d in glob.glob(os.path.join(args.src, "**", "*"), recursive=True)
        if os.path.isdir(d) and _find_volume(d, args.modality) is not None
    )
    if not subject_dirs:
        sys.exit(f"[error] no subject folders with *{args.modality}.nii* under {args.src}")

    resize = v2.Resize((args.image_size, args.image_size), antialias=True)

    imgs, grades, sids = [], [], []
    skipped = 0
    for sdir in tqdm(subject_dirs, desc="subjects"):
        sid = os.path.basename(sdir)
        if sid not in grade_map:
            skipped += 1
            continue
        grade = grade_map[sid]

        vol_path = _find_volume(sdir, args.modality)
        seg_path = _find_volume(sdir, "seg")
        vol = nib.load(vol_path).get_fdata()  # (H, W, D)

        # Pick slices with the most tumor; fall back to brightest if no seg.
        if seg_path is not None:
            seg = nib.load(seg_path).get_fdata()
            tumor_area = (seg > 0).sum(axis=(0, 1))
        else:
            tumor_area = (vol > vol.mean()).sum(axis=(0, 1))
        if tumor_area.max() == 0:
            skipped += 1
            continue
        top = np.argsort(tumor_area)[::-1][: args.slices_per_subject]

        for z in top:
            if tumor_area[z] == 0:
                continue
            u8 = _to_uint8(vol[:, :, int(z)])                 # (H, W)
            t = torch.from_numpy(u8).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
            t = resize(t)
            imgs.append(t.to(torch.uint8))
            grades.append(grade)
            sids.append(sid)

    if not imgs:
        sys.exit("[error] no slices extracted — check --src / --modality.")

    images = torch.stack(imgs, dim=0)  # (N, 3, H, W) uint8
    torch.save(
        {
            "images": images,
            "grades": np.array(grades, dtype=np.int64),
            "subject_ids": sids,
            "image_size": args.image_size,
            "modality": args.modality,
        },
        out,
    )
    n_hgg = int(np.sum(grades))
    print(f"[done] {out}  slices={len(imgs)} (HGG={n_hgg}, LGG={len(imgs)-n_hgg}) "
          f"from {len(subject_dirs)-skipped} subjects; skipped={skipped}; "
          f"{os.path.getsize(out)/1e9:.2f} GB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="path to BraTS training folder")
    p.add_argument("--data-root", default="./data_cache")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--modality", default="flair",
                   help="MRI modality suffix: flair | t1ce | t1 | t2")
    p.add_argument("--slices-per-subject", type=int, default=5)
    p.add_argument("--force", action="store_true")
    main(p.parse_args())

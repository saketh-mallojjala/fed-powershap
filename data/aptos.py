"""APTOS 2019 Blindness Detection dataset.

Expects the standard Kaggle layout under ``{data_root}/aptos2019/``:

    data_root/aptos2019/
        train.csv             (id_code,diagnosis)
        train_images/         (PNGs named {id_code}.png)
        cache_<size>.pt       (optional; produced by scripts/prep_aptos_cache.py)

Kaggle's public test split has no labels, so we re-split ``train.csv``
stratified by diagnosis into our own train/test for federated evaluation.
The split is deterministic given a seed.

If a tensor cache file is present at ``cache_<image_size>.pt`` the dataset
loads from it instead of re-decoding PNGs every batch — typically 5-10x
faster on Mac.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


APTOS_NUM_CLASSES = 5  # 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR


class APTOSDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        df: pd.DataFrame,
        transform=None,
        cache: Optional[Dict[str, Any]] = None,
    ):
        self.image_dir = image_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.targets = self.df["diagnosis"].astype(int).tolist()
        # cache = {"images": uint8 tensor [N, 3, H, W], "id_to_idx": {code: i}}
        self.cache = cache

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if self.cache is not None:
            cache_idx = self.cache["id_to_idx"][row["id_code"]]
            img = self.cache["images"][cache_idx]            # uint8 [3, H, W]
        else:
            path = os.path.join(self.image_dir, f"{row['id_code']}.png")
            img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["diagnosis"])


def _stratified_split(
    df: pd.DataFrame,
    test_frac: float,
    num_classes: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    test_idx = []
    for c in range(num_classes):
        cls_idx = df.index[df["diagnosis"] == c].to_numpy().copy()
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(test_frac * len(cls_idx))))
        test_idx.extend(cls_idx[:n_test].tolist())
    test_mask = df.index.isin(test_idx)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy()


def _maybe_load_cache(base: str, image_size: int) -> Optional[Dict[str, Any]]:
    cache_path = os.path.join(base, f"cache_{image_size}.pt")
    if not os.path.exists(cache_path):
        return None
    print(f"[aptos] using tensor cache: {cache_path}")
    blob = torch.load(cache_path, weights_only=False)
    id_to_idx = {code: i for i, code in enumerate(blob["id_codes"])}
    return {"images": blob["images"], "id_to_idx": id_to_idx}


def load_aptos(
    data_root: str,
    train_tf,
    test_tf,
    test_frac: float = 0.2,
    seed: int = 42,
    image_size: int = 224,
) -> Tuple[APTOSDataset, APTOSDataset]:
    base = os.path.join(data_root, "aptos2019")
    csv_path = os.path.join(base, "train.csv")
    img_dir = os.path.join(base, "train_images")

    if not os.path.exists(csv_path) or not os.path.isdir(img_dir):
        raise FileNotFoundError(
            "APTOS 2019 not found. Expected layout:\n"
            f"  {csv_path}\n"
            f"  {img_dir}/<id_code>.png\n\n"
            "Run: python scripts/download_aptos.py\n"
            "(requires Kaggle API credentials at ~/.kaggle/access_token or kaggle.json)"
        )

    cache = _maybe_load_cache(base, image_size)
    df = pd.read_csv(csv_path)
    train_df, test_df = _stratified_split(df, test_frac, APTOS_NUM_CLASSES, seed)
    train = APTOSDataset(img_dir, train_df, train_tf, cache=cache)
    test = APTOSDataset(img_dir, test_df, test_tf, cache=cache)
    return train, test

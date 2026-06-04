"""BraTS reframed as 2D tumor-grade (HGG vs LGG) classification.

BraTS ships 3D multi-modal MRI volumes for *segmentation*. To slot it into this
classification FL pipeline we reframe it as binary **HGG (high-grade glioma) vs
LGG (low-grade glioma)** classification on representative 2D axial slices.

Because decoding NIfTI volumes and choosing tumor-bearing slices is slow, the
heavy lifting happens once in ``scripts/prep_brats_slices.py``, which writes a
tensor cache:

    data_root/brats/cache_<image_size>.pt
        images:      uint8 tensor [N, 3, H, W]   (a modality replicated to 3ch)
        grades:      int8/array  [N]             (0=LGG, 1=HGG)
        subject_ids: list[str]   [N]             (for subject-level splitting)

This loader only reads that cache, splits **by subject** (no subject appears in
both train and test, avoiding slice leakage), and returns datasets that expose
``.targets`` like the other loaders.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


BRATS_NUM_CLASSES = 2  # 0 = LGG, 1 = HGG


class BraTSSliceDataset(Dataset):
    def __init__(self, images: torch.Tensor, grades: np.ndarray, transform=None):
        self.images = images                 # uint8 [N, 3, H, W]
        self.targets = grades.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        img = self.images[idx]               # uint8 [3, H, W]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[idx])


def _subject_split(
    subject_ids: np.ndarray,
    grades: np.ndarray,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split slice indices into train/test by *subject*, stratified by grade."""
    rng = np.random.default_rng(seed)
    # Grade per subject (subjects are single-grade in BraTS).
    subjects = {}
    for sid, g in zip(subject_ids, grades):
        subjects[sid] = int(g)
    uniq = np.array(list(subjects.keys()))
    uniq_grades = np.array([subjects[s] for s in uniq])

    test_subjects = set()
    for g in (0, 1):
        sg = uniq[uniq_grades == g]
        rng.shuffle(sg)
        n_test = max(1, int(round(test_frac * len(sg))))
        test_subjects.update(sg[:n_test].tolist())

    is_test = np.array([sid in test_subjects for sid in subject_ids])
    return np.where(~is_test)[0], np.where(is_test)[0]


def load_brats(
    data_root: str,
    train_tf,
    test_tf,
    test_frac: float = 0.2,
    seed: int = 42,
    image_size: int = 224,
) -> Tuple[BraTSSliceDataset, BraTSSliceDataset]:
    base = os.path.join(data_root, "brats")
    cache_path = os.path.join(base, f"cache_{image_size}.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            "BraTS slice cache not found. Expected:\n"
            f"  {cache_path}\n\n"
            "Build it from the raw BraTS volumes with:\n"
            f"  python scripts/prep_brats_slices.py --image-size {image_size} "
            "--src <path/to/BraTS/training>"
        )

    print(f"[brats] using slice cache: {cache_path}")
    blob = torch.load(cache_path, weights_only=False)
    images = blob["images"]                       # uint8 [N, 3, H, W]
    grades = np.asarray(blob["grades"])
    subject_ids = np.asarray(blob["subject_ids"])

    train_idx, test_idx = _subject_split(subject_ids, grades, test_frac, seed)
    train = BraTSSliceDataset(images[train_idx], grades[train_idx], train_tf)
    test = BraTSSliceDataset(images[test_idx], grades[test_idx], test_tf)
    print(f"[brats] train slices={len(train)} test slices={len(test)} "
          f"(subject-level split)")
    return train, test

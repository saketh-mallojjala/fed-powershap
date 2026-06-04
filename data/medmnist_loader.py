"""Generalized MedMNIST loader for the federated pipeline.

Three of our target datasets are MedMNIST v2 collections, so a single loader
covers them all instead of one file per dataset:

    octmnist     -> OCTMNIST   (retinal OCT, 4 classes, grayscale, single-label)
    isic         -> DermaMNIST (HAM10000 / ISIC dermoscopy, 7 classes, RGB, single-label)
    chestxray14  -> ChestMNIST (NIH ChestX-ray14, 14 classes, grayscale, multi-label)

MedMNIST ships pre-defined train/val/test splits and (via MedMNIST+) source
resolutions of 28/64/128/224 px. We merge ``train``+``val`` into our training
pool and keep the official ``test`` split for evaluation. Everything is loaded
``as_rgb=True`` so the ImageNet-pretrained ResNet keeps its 3-channel input;
grayscale datasets are simply replicated across channels.

For multi-label ChestMNIST, ``__getitem__`` returns the full multi-hot float
vector (consumed by ``BCEWithLogitsLoss``), while ``.targets`` exposes the
``argmax`` label used *only* for Dirichlet partitioning / histograms / plots.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# name -> {medmnist flag, num_classes, task, gray}
MEDMNIST_REGISTRY = {
    "octmnist":    {"flag": "octmnist",   "num_classes": 4,  "task": "single_label", "gray": True},
    "isic":        {"flag": "dermamnist", "num_classes": 7,  "task": "single_label", "gray": False},
    "chestxray14": {"flag": "chestmnist", "num_classes": 14, "task": "multi_label",  "gray": True},
}

# Source resolutions MedMNIST+ actually ships.
_MEDMNIST_SIZES = (28, 64, 128, 224)


def _snap_size(image_size: int) -> int:
    """Snap a requested image size to the nearest available MedMNIST source size.

    The dataset transform resizes to ``image_size`` anyway; this only picks the
    resolution we download/decode from. We choose the smallest available size
    that is >= the request (capped at 224) to avoid needless upsampling.
    """
    for s in _MEDMNIST_SIZES:
        if image_size <= s:
            return s
    return _MEDMNIST_SIZES[-1]


class MedMNISTDataset(Dataset):
    """Wraps one or more underlying MedMNIST split datasets as a single dataset.

    The underlying datasets already apply the torchvision transform, so
    ``__getitem__`` returns the transformed image tensor plus the label
    (int for single-label, multi-hot float tensor for multi-label).
    """

    def __init__(self, bases: List[Dataset], task: str):
        self.bases = bases
        self.task = task
        self._lengths = [len(b) for b in bases]
        self._cum = np.cumsum([0] + self._lengths)

        labels = np.concatenate([np.asarray(b.labels) for b in bases], axis=0)
        if task == "multi_label":
            self.multihot = labels.astype(np.float32)             # (N, C)
            self.targets = labels.argmax(axis=1).astype(np.int64)  # partition label
        else:
            self.targets = labels.reshape(-1).astype(np.int64)
            self.multihot = None

    def __len__(self) -> int:
        return int(self._cum[-1])

    def _locate(self, idx: int) -> Tuple[int, int]:
        b = int(np.searchsorted(self._cum, idx, side="right") - 1)
        return b, idx - int(self._cum[b])

    def __getitem__(self, idx: int):
        b, j = self._locate(idx)
        img, _ = self.bases[b][j]  # transform already applied by the base dataset
        if self.task == "multi_label":
            return img, torch.from_numpy(self.multihot[idx])
        return img, int(self.targets[idx])


def load_medmnist(
    flag: str,
    data_root: str,
    train_tf,
    test_tf,
    task: str = "single_label",
    image_size: int = 224,
    seed: int = 42,
) -> Tuple[MedMNISTDataset, MedMNISTDataset]:
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "medmnist is required for octmnist/isic/chestxray14. "
            "Install with: pip install medmnist"
        ) from e

    if flag not in INFO:
        raise ValueError(f"Unknown MedMNIST flag: {flag}")

    DataClass = getattr(medmnist, INFO[flag]["python_class"])
    root = os.path.join(data_root, "medmnist")
    os.makedirs(root, exist_ok=True)

    src_size = _snap_size(image_size)
    common = dict(download=True, as_rgb=True, size=src_size, root=root)

    # train+val both get the (augmenting) train transform; they become our pool.
    train_base = DataClass(split="train", transform=train_tf, **common)
    val_base = DataClass(split="val", transform=train_tf, **common)
    test_base = DataClass(split="test", transform=test_tf, **common)

    train = MedMNISTDataset([train_base, val_base], task)
    test = MedMNISTDataset([test_base], task)
    print(
        f"[medmnist] {flag}: src_size={src_size} task={task} "
        f"train={len(train)} test={len(test)}"
    )
    return train, test

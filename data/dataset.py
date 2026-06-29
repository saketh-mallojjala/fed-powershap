"""Dataset loading and Dirichlet non-IID partitioning for FL clients."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .aptos import load_aptos
from .brats import load_brats
from .medmnist_loader import MEDMNIST_REGISTRY, load_medmnist


def _get_transforms(dataset: str, image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset in ("mnist", "fmnist"):
        mean, std = (0.1307,), (0.3081,)
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = train_tf
    elif dataset == "aptos":
        # ImageNet stats — pretrained backbone expects them.
        # Use v2 transforms because they work on uint8 tensors (cache mode)
        # and PIL alike.
        from torchvision.transforms import v2
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_tf = v2.Compose([
            v2.Resize((image_size, image_size), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(20),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ])
        test_tf = v2.Compose([
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ])
    elif dataset in MEDMNIST_REGISTRY or dataset == "brats":
        # MedMNIST (loaded as_rgb) and BraTS slices (replicated to 3ch) both
        # arrive as 3-channel images; use ImageNet stats for the pretrained
        # backbone. v2 transforms handle PIL and uint8 tensors alike.
        from torchvision.transforms import v2
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_tf = v2.Compose([
            v2.Resize((image_size, image_size), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(15),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ])
        test_tf = v2.Compose([
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return train_tf, test_tf


def _load_raw(dataset: str, root: str, image_size: int = 224, seed: int = 42):
    train_tf, test_tf = _get_transforms(dataset, image_size=image_size)
    os.makedirs(root, exist_ok=True)
    if dataset == "cifar10":
        train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    elif dataset == "mnist":
        train = datasets.MNIST(root, train=True, download=True, transform=train_tf)
        test = datasets.MNIST(root, train=False, download=True, transform=test_tf)
    elif dataset == "fmnist":
        train = datasets.FashionMNIST(root, train=True, download=True, transform=train_tf)
        test = datasets.FashionMNIST(root, train=False, download=True, transform=test_tf)
    elif dataset == "aptos":
        train, test = load_aptos(root, train_tf, test_tf, seed=seed, image_size=image_size)
    elif dataset in MEDMNIST_REGISTRY:
        entry = MEDMNIST_REGISTRY[dataset]
        train, test = load_medmnist(
            entry["flag"], root, train_tf, test_tf,
            task=entry["task"], image_size=image_size, seed=seed,
        )
    elif dataset == "brats":
        train, test = load_brats(root, train_tf, test_tf, seed=seed, image_size=image_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return train, test


class NoisyLabelDataset(Dataset):
    """Wrap a client subset and override a precomputed set of training labels
    (single-label). Simulates a low-quality client with mislabeled data."""

    def __init__(self, base: Dataset, remap: Dict[int, int]):
        self.base = base
        self.remap = remap

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        if i in self.remap:
            return x, self.remap[i]
        return x, y


def _load_train_eval_view(dataset: str, root: str, image_size: int, seed: int):
    """Return the *training* pool with **test** transforms (no augmentation).

    Used to build per-client local test sets for fairness evaluation: same
    underlying images as the client train subsets but evaluated deterministically.
    The train/test split inside each loader is seeded, so item ordering matches
    the augmenting view loaded by ``_load_raw``.
    """
    _, test_tf = _get_transforms(dataset, image_size=image_size)
    if dataset == "cifar10":
        return datasets.CIFAR10(root, train=True, download=False, transform=test_tf)
    if dataset == "mnist":
        return datasets.MNIST(root, train=True, download=False, transform=test_tf)
    if dataset == "fmnist":
        return datasets.FashionMNIST(root, train=True, download=False, transform=test_tf)
    if dataset == "aptos":
        train, _ = load_aptos(root, test_tf, test_tf, seed=seed, image_size=image_size)
        return train
    if dataset in MEDMNIST_REGISTRY:
        entry = MEDMNIST_REGISTRY[dataset]
        train, _ = load_medmnist(
            entry["flag"], root, test_tf, test_tf,
            task=entry["task"], image_size=image_size, seed=seed,
        )
        return train
    if dataset == "brats":
        train, _ = load_brats(root, test_tf, test_tf, seed=seed, image_size=image_size)
        return train
    raise ValueError(f"Unsupported dataset: {dataset}")


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    num_classes: int,
    seed: int = 42,
    min_size: int = 10,
) -> List[np.ndarray]:
    """Standard Dirichlet label-skew partition.

    Draws per-class proportions from Dirichlet(alpha) and distributes class indices
    to clients. Retries until every client has at least `min_size` samples.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    while True:
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet([alpha] * num_clients)
            splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx_c, splits)):
                client_indices[k].extend(part.tolist())
        if min(len(ci) for ci in client_indices) >= min_size:
            break
    return [np.array(ci, dtype=np.int64) for ci in client_indices]


def build_federated_datasets(
    cfg,
) -> Tuple[List[Subset], Dataset, Dict[int, np.ndarray], List[Optional[Subset]]]:
    """Return per-client train subsets, the shared test set, per-client label
    arrays, and per-client **local test** subsets (or ``None`` per client when
    ``local_test_frac == 0``).

    The per-client local test sets hold out ``cfg.local_test_frac`` of each
    client's data (seeded, so identical across methods) and use non-augmenting
    transforms. They drive the fairness (Jain index) evaluation.
    """
    train, test = _load_raw(
        cfg.dataset,
        cfg.data_root,
        image_size=getattr(cfg, "image_size", 224),
        seed=cfg.seed,
    )
    if hasattr(train, "targets"):
        labels = np.array(train.targets)
    else:
        labels = np.array([y for _, y in train])

    # Optionally cap the training pool (stratified) so large datasets stay
    # comparable in scale / per-round cost to the smaller ones. `keep` maps
    # capped-array positions back to original dataset indices.
    keep = np.arange(len(labels))
    cap = getattr(cfg, "max_train_samples", 0)
    if cap and cap < len(labels):
        keep = _stratified_subsample(labels, cap, cfg.num_classes, cfg.seed)
        labels = labels[keep]
        print(f"[data] capped train pool to {len(keep)} samples (stratified)")

    client_idx = dirichlet_partition(
        labels=labels,
        num_clients=cfg.num_clients,
        alpha=cfg.alpha,
        num_classes=cfg.num_classes,
        seed=cfg.seed,
    )

    frac = float(getattr(cfg, "local_test_frac", 0.0))
    train_eval = None
    if frac > 0.0:
        train_eval = _load_train_eval_view(
            cfg.dataset, cfg.data_root,
            image_size=getattr(cfg, "image_size", 224), seed=cfg.seed,
        )

    rng = np.random.default_rng(cfg.seed + 1)  # offset so split != partition rng
    client_subsets: List[Subset] = []
    client_test_subsets: List[Optional[Subset]] = []
    client_labels: Dict[int, np.ndarray] = {}
    for k, idx in enumerate(client_idx):
        orig = keep[idx]                         # original dataset indices
        n = len(idx)
        n_test = int(round(n * frac)) if frac > 0.0 else 0
        if frac > 0.0 and n > 1:
            n_test = min(max(n_test, 1), n - 1)  # keep >=1 train and >=1 test
        else:
            n_test = 0
        perm = rng.permutation(n)
        test_pos, train_pos = perm[:n_test], perm[n_test:]

        client_subsets.append(Subset(train, orig[train_pos].tolist()))
        client_labels[k] = labels[idx][train_pos]
        if n_test > 0 and train_eval is not None:
            client_test_subsets.append(Subset(train_eval, orig[test_pos].tolist()))
        else:
            client_test_subsets.append(None)

    # Optionally corrupt a fraction of clients' TRAINING labels (single-label).
    noisy_frac = float(getattr(cfg, "noisy_client_frac", 0.0))
    noise_rate = float(getattr(cfg, "label_noise_rate", 0.0))
    if noisy_frac > 0.0 and noise_rate > 0.0 and getattr(cfg, "task", "single_label") == "single_label":
        nrng = np.random.default_rng(cfg.seed + 2)
        n_noisy = max(1, int(round(cfg.num_clients * noisy_frac)))
        noisy_ids = nrng.choice(cfg.num_clients, size=n_noisy, replace=False)
        for k in noisy_ids:
            ck = client_labels[k]
            m = len(ck)
            n_flip = int(round(m * noise_rate))
            if n_flip == 0:
                continue
            flip_pos = nrng.choice(m, size=n_flip, replace=False)
            remap = {}
            for p in flip_pos:
                true = int(ck[p])
                choices = [c for c in range(cfg.num_classes) if c != true]
                remap[int(p)] = int(nrng.choice(choices))
            client_subsets[k] = NoisyLabelDataset(client_subsets[k], remap)
        print(f"[data] injected label noise: clients {sorted(noisy_ids.tolist())} "
              f"@ rate {noise_rate}")

    return client_subsets, test, client_labels, client_test_subsets


def _stratified_subsample(
    labels: np.ndarray, max_n: int, num_classes: int, seed: int
) -> np.ndarray:
    """Pick ~max_n indices, keeping each class's proportion of the whole."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    keep: List[int] = []
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        take = int(round(len(idx_c) * max_n / n))
        take = min(len(idx_c), max(1, take)) if len(idx_c) else 0
        if take:
            keep.extend(rng.choice(idx_c, size=take, replace=False).tolist())
    return np.sort(np.array(keep, dtype=np.int64))


def get_test_loader(test_set: Dataset, batch_size: int = 256, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=False,  # MPS doesn't support pinned memory
    )


def make_client_loader(
    subset: Subset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    g = torch.Generator()
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=g,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )

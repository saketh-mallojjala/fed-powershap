"""Dataset loading and Dirichlet non-IID partitioning for FL clients."""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def _get_transforms(dataset: str) -> Tuple[transforms.Compose, transforms.Compose]:
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
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return train_tf, test_tf


def _load_raw(dataset: str, root: str):
    train_tf, test_tf = _get_transforms(dataset)
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
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return train, test


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


def build_federated_datasets(cfg) -> Tuple[List[Subset], Dataset, Dict[int, np.ndarray]]:
    """Return per-client train subsets, the shared test set, and per-client label arrays."""
    train, test = _load_raw(cfg.dataset, cfg.data_root)
    if hasattr(train, "targets"):
        labels = np.array(train.targets)
    else:
        labels = np.array([y for _, y in train])

    client_idx = dirichlet_partition(
        labels=labels,
        num_clients=cfg.num_clients,
        alpha=cfg.alpha,
        num_classes=cfg.num_classes,
        seed=cfg.seed,
    )

    client_subsets = [Subset(train, idx.tolist()) for idx in client_idx]
    client_labels = {k: labels[idx] for k, idx in enumerate(client_idx)}
    return client_subsets, test, client_labels


def get_test_loader(test_set: Dataset, batch_size: int = 256) -> DataLoader:
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


def make_client_loader(subset: Subset, batch_size: int, shuffle: bool = True) -> DataLoader:
    g = torch.Generator()
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=g)

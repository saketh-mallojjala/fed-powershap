"""Tabular mental-health survey loaders (Kaggle depression-screening datasets).

Why this module exists: the FedPowerShap harness only loaded image datasets, so
the incentive/mental-health case study (docs/INCENTIVE_PROPOSAL.md §9) needs a
new code path. The output is deliberately shaped like the image loaders — a
Dataset exposing ``.targets`` — so ``dataset.build_federated_datasets`` (Dirichlet
partition, per-client local test split, label-noise injection) works unchanged.

Two design points that matter for the study:

1. **Feature regime.** On the Student Depression dataset the item
   *"Have you ever had suicidal thoughts?"* is the single strongest predictor
   (SHAP analyses rank it first) and is a core diagnostic symptom rather than an
   independent risk factor. Including it pushes the task close to saturation, and
   a saturated task cannot discriminate between aggregators. ``feature_regime``
   selects ``"full"`` (keep it) or ``"screening"`` (drop it — the headline regime).

2. **Leakage-safe scaling.** The train/test split happens *before* the scaler is
   fit, and the scaler sees only the training pool. Note this is a single global
   scaler shared by all clients, i.e. a mild centralisation assumption standard in
   tabular FL work; it is recorded here rather than hidden.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Ordinal encodings — these columns are categorical strings with a real order,
# so one-hot would throw away the ordering the MLP can exploit.
_ORDINAL_MAPS: Dict[str, Dict[str, float]] = {
    "Sleep Duration": {
        "Less than 5 hours": 0.0,
        "5-6 hours": 1.0,
        "7-8 hours": 2.0,
        "More than 8 hours": 3.0,
        # "Others" (18 rows) has no defined position -> NaN -> row dropped.
    },
    "Dietary Habits": {"Unhealthy": 0.0, "Moderate": 1.0, "Healthy": 2.0},
}

_BINARY_MAPS: Dict[str, Dict[str, float]] = {
    "Gender": {"Male": 1.0, "Female": 0.0},
    "Have you ever had suicidal thoughts ?": {"Yes": 1.0, "No": 0.0},
    "Family History of Mental Illness": {"Yes": 1.0, "No": 0.0},
}

# The feature that makes the task near-saturated; removed in the screening regime.
SUICIDAL_IDEATION_COL = "Have you ever had suicidal thoughts ?"

TABULAR_REGISTRY: Dict[str, dict] = {
    # 27,901 rows. Primary set. `Profession` is 99.9% "Student" and
    # `Work Pressure` / `Job Satisfaction` are 99.97% zero -> degenerate, dropped.
    "student_depression": dict(
        csv="Student Depression Dataset.csv",
        kaggle="hopesb/student-depression-dataset",
        target="Depression",
        drop=["id", "Profession", "Work Pressure", "Job Satisfaction"],
        onehot=["City", "Degree"],
    ),
    # 502 rows. A genuinely small cohort; balanced target (50.2% positive).
    "depression_student": dict(
        csv="Depression Student Dataset.csv",
        kaggle="ikynahidwin/depression-student-dataset",
        target="Depression",
        drop=[],
        onehot=[],
    ),
}


@dataclass
class TabularMeta:
    """Everything downstream code (and the write-up) needs about a built split."""
    num_features: int
    num_classes: int
    feature_names: List[str]
    n_train: int
    n_test: int
    positive_rate: float
    feature_regime: str
    dropped_rows: int


class TabularDataset(Dataset):
    """Dense float32 features + int64 labels, exposing ``.targets``.

    ``build_federated_datasets`` reads ``.targets`` to build the Dirichlet
    partition, exactly as it does for torchvision datasets.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        self.targets = self.y.tolist()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], int(self.y[i])


def _resolve_csv(root: str, spec: dict) -> str:
    path = os.path.join(root, "mentalhealth", spec["csv"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download it with:\n"
            f"  export KAGGLE_CONFIG_DIR=$(pwd)\n"
            f"  kaggle datasets download -d {spec['kaggle']} "
            f"-p {os.path.join(root, 'mentalhealth')} --unzip"
        )
    return path


def _encode(df, spec: dict, feature_regime: str):
    """CSV -> (X float array, y int array, feature names, rows dropped)."""
    import pandas as pd

    target = spec["target"]
    df = df.drop(columns=[c for c in spec["drop"] if c in df.columns])

    if feature_regime == "screening" and SUICIDAL_IDEATION_COL in df.columns:
        df = df.drop(columns=[SUICIDAL_IDEATION_COL])
    elif feature_regime not in ("full", "screening"):
        raise ValueError(f"feature_regime must be 'full' or 'screening', got {feature_regime!r}")

    # Target -> {0,1}. Handles both the int (1/0) and string (Yes/No) encodings.
    y_raw = df[target]
    if y_raw.dtype == object:
        y = y_raw.map({"Yes": 1, "No": 0})
    else:
        y = y_raw.astype("float64")
    df = df.drop(columns=[target])

    n_before = len(df)
    blocks: List[np.ndarray] = []
    names: List[str] = []
    dense_cols: List[str] = []          # columns to standardize (not one-hot)

    for col in df.columns:
        if col in _ORDINAL_MAPS:
            s = df[col].map(_ORDINAL_MAPS[col])
        elif col in _BINARY_MAPS:
            s = df[col].map(_BINARY_MAPS[col])
        elif col in spec["onehot"]:
            dummies = pd.get_dummies(df[col], prefix=col, dtype="float64")
            blocks.append(dummies.to_numpy())
            names.extend(list(dummies.columns))
            continue
        elif df[col].dtype == object:
            # Unexpected string column: one-hot if small, else drop loudly.
            if df[col].nunique() <= 2:
                vals = sorted(df[col].dropna().unique())
                s = df[col].map({vals[0]: 0.0, vals[-1]: 1.0})
            else:
                dummies = pd.get_dummies(df[col], prefix=col, dtype="float64")
                blocks.append(dummies.to_numpy())
                names.extend(list(dummies.columns))
                continue
        else:
            s = df[col].astype("float64")
        blocks.append(s.to_numpy().reshape(-1, 1))
        names.append(col)
        dense_cols.append(col)

    X = np.concatenate(blocks, axis=1)
    yv = y.to_numpy(dtype="float64")

    # Drop rows with any NaN (from unmapped ordinals, e.g. "Others", or the 3
    # missing Financial Stress values).
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(yv))
    X, yv = X[ok], yv[ok]
    dense_mask = np.array([n in dense_cols for n in names], dtype=bool)
    return X, yv.astype(np.int64), names, n_before - int(ok.sum()), dense_mask


def load_tabular(
    name: str,
    root: str = "./data_cache",
    seed: int = 42,
    test_frac: float = 0.2,
    feature_regime: str = "screening",
) -> Tuple[TabularDataset, TabularDataset, TabularMeta]:
    """Load a tabular mental-health dataset as (train, test, meta).

    The split is stratified on the target and seeded, so it is identical across
    methods. The standardizer is fit on the training rows only.
    """
    import pandas as pd

    if name not in TABULAR_REGISTRY:
        raise ValueError(f"Unknown tabular dataset {name!r}. Choices: {sorted(TABULAR_REGISTRY)}")
    spec = TABULAR_REGISTRY[name]
    df = pd.read_csv(_resolve_csv(root, spec))

    X, y, names, dropped, dense_mask = _encode(df, spec, feature_regime)

    # Stratified train/test split.
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_test = int(round(len(idx_c) * test_frac))
        test_idx.extend(idx_c[:n_test].tolist())
        train_idx.extend(idx_c[n_test:].tolist())
    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))

    # Standardize dense (non one-hot) columns using TRAIN statistics only.
    mu = X[train_idx][:, dense_mask].mean(axis=0)
    sd = X[train_idx][:, dense_mask].std(axis=0)
    sd[sd < 1e-8] = 1.0
    X = X.copy()
    X[:, dense_mask] = (X[:, dense_mask] - mu) / sd

    train = TabularDataset(X[train_idx], y[train_idx])
    test = TabularDataset(X[test_idx], y[test_idx])
    meta = TabularMeta(
        num_features=X.shape[1],
        num_classes=int(len(np.unique(y))),
        feature_names=names,
        n_train=len(train_idx),
        n_test=len(test_idx),
        positive_rate=float(y.mean()),
        feature_regime=feature_regime,
        dropped_rows=dropped,
    )
    return train, test, meta

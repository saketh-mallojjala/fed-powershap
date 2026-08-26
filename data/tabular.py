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
    # ---- datasets with NON-SYNTHETIC client structure ----
    # Every result so far partitions clients with a Dirichlet draw over labels.
    # That is standard but it is not how real federations differ: real silos
    # differ by case mix and referral pathway, not by a symmetric label skew.
    # These two carry a real grouping variable, so clients can be formed from
    # the data itself rather than from an RNG.
    #
    # Diabetes 130-US Hospitals (UCI id=296), 101,766 encounters.
    # NOTE: despite the name, the public release carries NO hospital identifier
    # -- the 130 sites are de-identified away. `medical_specialty` is used
    # instead: 16 specialties hold >=500 encounters each and their 30-day
    # readmission rate ranges 4.8%-15.4%, i.e. a 3x spread that arises from the
    # data rather than from a Dirichlet parameter. This is an ATTRIBUTE-based
    # natural partition (departments), not a site-based one, and is described
    # that way in the write-up.
    "diabetes130": dict(
        uci_id=296,
        target="readmitted",
        positive="!NO",                 # ANY readmission (~46%). The "<30"
                                        # target was tried first and rejected at
                                        # the Step-0 headroom check: centralized
                                        # bacc 0.524 / AUC 0.629, below the
                                        # majority baseline on accuracy.
        drop=["weight", "payer_code", "encounter_id", "patient_nbr"],
        onehot=["race", "gender", "age", "admission_type_id",
                "discharge_disposition_id", "admission_source_id",
                "max_glu_serum", "A1Cresult", "change", "diabetesMed"],
        partition_col="medical_specialty",
        min_client_rows=500,
        icd_cols=["diag_1", "diag_2", "diag_3"],
        impute=True,
    ),
    # UCI Heart Disease -- FOUR genuinely separate hospitals:
    # Cleveland (303), Hungarian (294), Switzerland (123), Long Beach VA (200).
    # Small, but the client boundary is a real institution in a different
    # country, which is the strongest natural-partition evidence available
    # without a data-use agreement. Same data underlying FLamby's
    # Fed-Heart-Disease benchmark.
    "heart_disease": dict(
        uci_multisite=["cleveland", "hungarian", "switzerland", "va"],
        target="num",
        positive=">0",                  # any degree of disease = positive
        drop=[],
        onehot=["cp", "restecg", "slope", "thal"],
        partition_col="__site__",
        min_client_rows=50,
        impute=True,
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
    # Natural client label per TRAIN row (None => use Dirichlet).
    train_groups: Optional[List[str]] = None


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


_HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
_HEART_BASE = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "heart-disease/processed.{}.data")


def _load_uci(root: str, spec: dict):
    """Fetch a UCI dataset, cached locally so runs are offline-repeatable.

    Returns a dataframe that still carries the partition column, because client
    structure here comes from the data rather than from an RNG.
    """
    import pandas as pd

    cache_dir = os.path.join(root, "uci")
    os.makedirs(cache_dir, exist_ok=True)

    if "uci_multisite" in spec:                     # heart disease: 4 hospitals
        cache = os.path.join(cache_dir, "heart_disease_4site.csv")
        if os.path.exists(cache):
            return pd.read_csv(cache)
        frames = []
        for site in spec["uci_multisite"]:
            df = pd.read_csv(_HEART_BASE.format(site), header=None,
                             names=_HEART_COLS, na_values="?")
            df["__site__"] = site                   # the natural client label
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(cache, index=False)
        return out

    cache = os.path.join(cache_dir, f"uci_{spec['uci_id']}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, low_memory=False)
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=spec["uci_id"])
    out = pd.concat([d.data.features, d.data.targets], axis=1)
    out.to_csv(cache, index=False)
    return out


_ICD_CHAPTERS = [
    (1, 139, "infectious"), (140, 239, "neoplasm"), (240, 279, "endocrine"),
    (280, 289, "blood"), (290, 319, "mental"), (320, 389, "nervous"),
    (390, 459, "circulatory"), (460, 519, "respiratory"), (520, 579, "digestive"),
    (580, 629, "genitourinary"), (630, 679, "pregnancy"), (680, 709, "skin"),
    (710, 739, "musculoskeletal"), (740, 759, "congenital"), (760, 779, "perinatal"),
    (780, 799, "illdefined"), (800, 999, "injury"),
]


def _icd_chapter(code) -> str:
    """Map an ICD-9 code to its chapter.

    diag_1/2/3 carry ~700 distinct codes each; one-hot encoding them directly
    produced 2,404 features on 38k rows. Chapters are the grouping used
    throughout the literature on this dataset and keep the dimensionality sane.
    """
    s = str(code)
    if s.startswith(("V", "v")):
        return "supplementary"
    if s.startswith(("E", "e")):
        return "external"
    try:
        v = float(s)
    except ValueError:
        return "unknown"
    for lo, hi, name in _ICD_CHAPTERS:
        if lo <= v <= hi:
            return name
    return "unknown"


def _binarize(y_raw, positive: str):
    """Map a multi-valued clinical target to {0,1} under a stated rule."""
    if positive == ">0":                            # any disease grade counts
        return (y_raw.astype("float64") > 0).astype("float64")
    if positive.startswith("!"):                    # anything except this value
        return (y_raw.astype(str) != positive[1:]).astype("float64")
    return (y_raw.astype(str) == positive).astype("float64")


def _encode(df, spec: dict, feature_regime: str):
    """CSV -> (X float array, y int array, feature names, rows dropped)."""
    import pandas as pd

    target = spec["target"]
    df = df.drop(columns=[c for c in spec["drop"] if c in df.columns])

    if feature_regime == "screening" and SUICIDAL_IDEATION_COL in df.columns:
        df = df.drop(columns=[SUICIDAL_IDEATION_COL])
    elif feature_regime not in ("full", "screening"):
        raise ValueError(f"feature_regime must be 'full' or 'screening', got {feature_regime!r}")

    # Target -> {0,1}. Handles int (1/0), string (Yes/No), and the clinical
    # multi-valued encodings declared via spec["positive"].
    y_raw = df[target]
    if "positive" in spec:
        y = _binarize(y_raw, spec["positive"])
    elif y_raw.dtype == object:
        y = y_raw.map({"Yes": 1, "No": 0})
    else:
        y = y_raw.astype("float64")
    df = df.drop(columns=[target])

    # Hold the natural-partition column aside: it labels the client, it is not
    # a feature. Leaving it in would leak the client identity into the model.
    groups = None
    pcol = spec.get("partition_col")
    if pcol and pcol in df.columns:
        groups = df[pcol].astype(str).fillna("unknown").to_numpy()
        df = df.drop(columns=[pcol])

    for c in spec.get("icd_cols", []):
        if c in df.columns:
            df[c] = df[c].map(_icd_chapter)

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
    # Multi-site clinical data is missing-heavy and missingness is NOT uniform
    # across sites: on heart-disease, ca/thal are largely absent outside
    # Cleveland, so dropping incomplete rows silently deleted 3 of the 4
    # hospitals. Impute instead, and drop rows only on a missing TARGET.
    if spec.get("impute"):
        col_med = np.nanmedian(X, axis=0)
        col_med = np.where(np.isnan(col_med), 0.0, col_med)
        nan_at = np.isnan(X)
        X = np.where(nan_at, col_med, X)

    ok = ~(np.isnan(X).any(axis=1) | np.isnan(yv))
    X, yv = X[ok], yv[ok]
    if groups is not None:
        groups = groups[ok]
    dense_mask = np.array([n in dense_cols for n in names], dtype=bool)
    return X, yv.astype(np.int64), names, n_before - int(ok.sum()), dense_mask, groups


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
    if "uci_id" in spec or "uci_multisite" in spec:
        df = _load_uci(root, spec)
    else:
        df = pd.read_csv(_resolve_csv(root, spec))

    X, y, names, dropped, dense_mask, groups = _encode(df, spec, feature_regime)
    # Keep only groups large enough to be a client; the long tail of tiny
    # specialties cannot support a local train/test split.
    if groups is not None:
        import collections
        counts = collections.Counter(groups)
        keep_g = {g for g, c in counts.items()
                  if c >= spec.get("min_client_rows", 100) and g not in ("?", "nan", "unknown")}
        m = np.array([g in keep_g for g in groups])
        X, y, groups = X[m], y[m], groups[m]
        print(f"[data] natural partition on {spec['partition_col']}: "
              f"{len(keep_g)} clients, {int(m.sum())} rows kept of {len(m)}")

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
        train_groups=(groups[train_idx].tolist() if groups is not None else None),
    )
    return train, test, meta

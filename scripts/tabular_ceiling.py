#!/usr/bin/env python3
"""Centralized (non-federated) ceiling on the tabular mental-health datasets.

Run this BEFORE any federated sweep. Its job is to answer one question:
**is there discriminative headroom for the FL comparison at all?**

If the pooled-data ceiling sits at the majority-class rate, no aggregator can
show a difference and the FL numbers are meaningless. If the ceiling is very high
(the "full" feature regime, where the suicidal-ideation item nearly determines the
label), the task is saturated and again nothing can be distinguished. The gap
between the majority baseline and the ceiling is the budget the FL experiment has
to work with.

Usage:
    python3 scripts/tabular_ceiling.py
    python3 scripts/tabular_ceiling.py --datasets student_depression --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tabular import load_tabular  # noqa: E402
from models.mlp import TabularMLP  # noqa: E402


def _metrics(logits: torch.Tensor, y: torch.Tensor) -> dict:
    prob = torch.softmax(logits, dim=1)[:, 1].numpy()
    pred = logits.argmax(dim=1).numpy()
    yn = y.numpy()
    acc = float((pred == yn).mean())
    # Balanced accuracy = mean per-class recall; the honest headline when the
    # base rate is 58.6% positive.
    recalls = [float((pred[yn == c] == c).mean()) for c in (0, 1) if (yn == c).any()]
    bacc = float(np.mean(recalls))
    # AUC via the rank identity (no sklearn dependency).
    order = np.argsort(prob)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(prob) + 1)
    n_pos, n_neg = int((yn == 1).sum()), int((yn == 0).sum())
    auc = float((ranks[yn == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return {"acc": acc, "balanced_acc": bacc, "auc": auc}


def train_central(name: str, regime: str, seed: int, epochs: int = 30,
                  hidden: int = 64, lr: float = 1e-3, bs: int = 128) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, test, meta = load_tabular(name, root="./data_cache", seed=seed,
                                     feature_regime=regime)
    model = TabularMLP(meta.num_features, meta.num_classes, hidden_dim=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    Xtr, ytr = train.X, train.y
    n = len(ytr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        out = _metrics(model(test.X), test.y)
    out["majority_baseline"] = float(max(meta.positive_rate, 1 - meta.positive_rate))
    out["num_features"] = meta.num_features
    out["n_train"], out["n_test"] = meta.n_train, meta.n_test
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["student_depression", "depression_student"])
    p.add_argument("--regimes", nargs="+", default=["screening", "full"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out",
                   default="studies/tabular_mentalhealth/tables/tabular_ceiling.json")
    args = p.parse_args()

    rows = []
    for ds in args.datasets:
        for regime in args.regimes:
            per_seed = [train_central(ds, regime, s, epochs=args.epochs)
                        for s in args.seeds]
            agg = {
                k: (float(np.mean([r[k] for r in per_seed])),
                    float(np.std([r[k] for r in per_seed])))
                for k in ("acc", "balanced_acc", "auc")
            }
            row = dict(dataset=ds, regime=regime, seeds=args.seeds,
                       majority=per_seed[0]["majority_baseline"],
                       num_features=per_seed[0]["num_features"],
                       n_train=per_seed[0]["n_train"], n_test=per_seed[0]["n_test"],
                       **{k: v[0] for k, v in agg.items()},
                       **{f"{k}_std": v[1] for k, v in agg.items()})
            rows.append(row)
            print(f"{ds:22s} {regime:10s} feats={row['num_features']:3d}  "
                  f"acc={row['acc']:.4f}±{row['acc_std']:.4f}  "
                  f"bacc={row['balanced_acc']:.4f}  auc={row['auc']:.4f}  "
                  f"(majority={row['majority']:.4f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n--- headroom (balanced acc above majority-class rate) ---")
    for r in rows:
        print(f"  {r['dataset']:22s} {r['regime']:10s} "
              f"{r['balanced_acc'] - 0.5:+.4f} above chance-balanced (0.50); "
              f"acc {r['acc'] - r['majority']:+.4f} above majority")


if __name__ == "__main__":
    main()

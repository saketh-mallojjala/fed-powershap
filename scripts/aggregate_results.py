"""Aggregate run logs into paper-ready tables (mean +/- std across seeds).

Parses ``logs/{dataset}_{method}_seed{n}-*.jsonl`` and reports, per
(dataset, method): final-round Accuracy, QWK (aptos) / AUC (chestxray14),
Jain fairness index, worst-client accuracy. Emits a CSV and a Markdown table.

Usage:
    python scripts/aggregate_results.py                       # all logs/
    python scripts/aggregate_results.py --datasets octmnist isic
    python scripts/aggregate_results.py --metric-rounds last  # or 'best'
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean, pstdev

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Preferred method display order / labels.
METHOD_ORDER = [
    "fedavg", "fedprox", "scaffold", "feddyn", "fedbn", "moon",
    "poc_fedavg", "fedce", "proposed",
]
METHOD_LABEL = {
    "fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD",
    "feddyn": "FedDyn", "fedbn": "FedBN", "moon": "MOON",
    "poc_fedavg": "PoC-FedAvg", "fedce": "FedCE", "proposed": "Proposed",
}
# {dataset: headline secondary metric key}
SECONDARY = {"aptos": "qwk", "chestxray14": "auc"}

# Anchor the method to a known name (alternation, longest first) so multi-
# underscore methods like 'poc_fedavg' don't get split into the dataset.
_METHODS_RE = "|".join(sorted(METHOD_ORDER, key=len, reverse=True))
_NAME_RE = re.compile(
    rf"^(?P<dataset>.+?)_(?P<method>{_METHODS_RE})_seed(?P<seed>\d+)-\d{{8}}-\d{{6}}\.jsonl$"
)


def parse_log(path):
    """Return (dataset, method, seed, evals[list of eval dicts]) or None."""
    base = os.path.basename(path)
    m = _NAME_RE.match(base)
    if not m:
        return None
    evals = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "eval":
                evals.append(rec)
    if not evals:
        return None
    return m["dataset"], m["method"], int(m["seed"]), evals


def pick_eval(evals, mode, key):
    """Pick the 'last' eval, or the 'best' by ``key`` (defaults to acc)."""
    if mode == "best":
        k = key if key and any(key in e for e in evals) else "acc"
        return max(evals, key=lambda e: e.get(k, e.get("acc", 0.0)))
    return evals[-1]


def fmt(values):
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{mean(values):.3f}±{pstdev(values):.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--metric-rounds", choices=["last", "best"], default="best")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # collect: data[dataset][method][metric] -> list over seeds
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(glob.glob(os.path.join(args.log_dir, "*_seed*-*.jsonl"))):
        parsed = parse_log(path)
        if not parsed:
            continue
        ds, method, seed, evals = parsed
        if args.datasets and ds not in args.datasets:
            continue
        sec = SECONDARY.get(ds)
        e = pick_eval(evals, args.metric_rounds, sec)
        data[ds][method]["acc"].append(e.get("acc", float("nan")))
        if sec and sec in e:
            data[ds][method][sec].append(e[sec])
        if "jain" in e:
            data[ds][method]["jain"].append(e["jain"])
        if "client_acc_min" in e:
            data[ds][method]["worst"].append(e["client_acc_min"])

    if not data:
        print(f"No matching logs in {args.log_dir} "
              f"(expected '{{dataset}}_{{method}}_seedN-*.jsonl').")
        return

    md_lines, csv_rows = [], []
    csv_rows.append(["dataset", "method", "n_seeds", "acc", "secondary_metric",
                     "secondary", "jain", "worst_client_acc"])

    for ds in sorted(data):
        sec = SECONDARY.get(ds, "")
        sec_hdr = sec.upper() if sec else "-"
        md_lines.append(f"\n### {ds}\n")
        md_lines.append(f"| Method | Accuracy | {sec_hdr} | Jain index | Worst-client acc |")
        md_lines.append("|---|---|---|---|---|")
        methods = [m for m in METHOD_ORDER if m in data[ds]]
        methods += [m for m in data[ds] if m not in METHOD_ORDER]
        for method in methods:
            d = data[ds][method]
            n = len(d["acc"])
            acc, jn, wc = fmt(d["acc"]), fmt(d.get("jain", [])), fmt(d.get("worst", []))
            secv = fmt(d.get(sec, [])) if sec else "-"
            label = METHOD_LABEL.get(method, method)
            star = " **" if method == "proposed" else ""
            md_lines.append(f"| {label}{star} | {acc} | {secv} | {jn} | {wc} |")
            csv_rows.append([ds, method, n, acc, sec, secv, jn, wc])

    md = "# Results (mean±std across seeds, metric=%s round)\n" % args.metric_rounds
    md += "\n".join(md_lines) + "\n"
    md_path = os.path.join(args.out_dir, "results_table.md")
    csv_path = os.path.join(args.out_dir, "results_table.csv")
    with open(md_path, "w") as f:
        f.write(md)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    print(md)
    print(f"[written] {md_path}\n[written] {csv_path}")


if __name__ == "__main__":
    main()

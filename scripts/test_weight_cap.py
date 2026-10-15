#!/usr/bin/env python3
"""Regression test for the aggregation weight cap (reviewer A, point 1).

The submitted manuscript's Eqs. (9)-(10) clip the preliminary weights at w_max
and then renormalise. Renormalising divides by a sum below 1, so it inflates
every surviving weight and the cap is not guaranteed. The reviewer is right.

This test pins three things:
  1. the manuscript formulation really does violate the cap (so the correction
     in the revision is warranted, not cosmetic);
  2. the previous water-filling implementation also violated it, because it
     looped a fixed len(w) times while proportional redistribution converges
     geometrically;
  3. the exact projection now in federated/shapley.py never violates it.
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from federated.shapley import _cap_weights


def paper_eq_9_10(a, wmax):
    c = np.minimum(a, wmax)
    return c / c.sum()


def old_waterfill(w, wmax, iters=None):
    w = w / w.sum()
    for _ in range(iters if iters is not None else len(w)):
        over = w > wmax + 1e-12
        if not over.any():
            break
        excess = (w[over] - wmax).sum()
        w[over] = wmax
        under = ~over
        if not under.any():
            break
        w[under] += excess * (w[under] / w[under].sum())
    return w / w.sum()


def main():
    fails = 0

    # --- 1. the worked example from the review -----------------------------
    a = np.array([0.50, 0.30, 0.10, 0.05, 0.05])
    got = _cap_weights(a.copy(), 0.30)
    want = np.array([0.30, 0.30, 0.20, 0.10, 0.10])
    ok = np.allclose(got, want, atol=1e-9)
    print(f"worked example        exact={got.round(4)}  {'PASS' if ok else 'FAIL'}")
    print(f"  manuscript Eqs(9-10)={paper_eq_9_10(a, 0.30).round(4)}  "
          f"max {paper_eq_9_10(a, 0.30).max():.4f} > 0.30")
    fails += not ok

    # --- 2. randomised stress ----------------------------------------------
    rng = np.random.default_rng(0)
    N = 200_000
    worst_paper = worst_old = worst_new = 0.0
    bad_paper = bad_old = bad_new = 0
    for _ in range(N):
        k = int(rng.integers(2, 11))
        a = rng.dirichlet(rng.uniform(0.05, 2.0, size=k))
        wmax = float(rng.uniform(1.0 / k + 1e-3, 1.0))   # feasible caps only
        for fn, nm in ((paper_eq_9_10, "paper"), (old_waterfill, "old"),
                       (_cap_weights, "new")):
            out = fn(a.copy(), wmax)
            v = out.max() - wmax
            if nm == "paper":
                worst_paper = max(worst_paper, v); bad_paper += v > 1e-9
            elif nm == "old":
                worst_old = max(worst_old, v); bad_old += v > 1e-9
            else:
                worst_new = max(worst_new, v); bad_new += v > 1e-9
                if abs(out.sum() - 1.0) > 1e-9 or (out < -1e-12).any():
                    fails += 1

    print(f"\n{N:,} random feasible cases")
    print(f"  manuscript Eqs (9)-(10) : {bad_paper:>7,} violations, worst {worst_paper:+.4f}")
    print(f"  previous water-filling  : {bad_old:>7,} violations, worst {worst_old:+.4f}")
    print(f"  exact projection (now)  : {bad_new:>7,} violations, worst {worst_new:+.2e}")
    fails += bad_new > 0

    # --- 3. edge cases ------------------------------------------------------
    edge = [
        (np.array([1.0]), 0.5, "single client"),
        (np.array([0.5, 0.5]), 0.4, "cap below uniform -> uniform"),
        (np.array([1.0, 0.0, 0.0]), 0.5, "zeros present"),
        (np.ones(7) / 7, 0.9, "already feasible, unchanged"),
    ]
    print()
    for a, wmax, name in edge:
        o = _cap_weights(a.copy(), wmax)
        ok = (abs(o.sum() - 1) < 1e-9) and (o.max() <= max(wmax, 1.0 / len(a)) + 1e-9)
        print(f"  {name:<34} {o.round(4)}  {'PASS' if ok else 'FAIL'}")
        fails += not ok

    print("\n" + ("ALL PASS" if fails == 0 else f"{fails} FAILURES"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

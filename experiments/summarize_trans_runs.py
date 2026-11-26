"""Summary of the referee-requested runs X1-X6 (docs/ieee_trans_review.md section 7), from outputs/.
Intervals: paired bootstrap 95% on reciprocal rank (10,000 resamples, seed 0).

Run:  python -m experiments.summarize_trans_runs
"""
import glob
import json

import numpy as np

from experiments.benchmark_localization import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs"


def load(pattern):
    return [json.loads(line) for f in sorted(glob.glob(str(OUT / pattern))) for line in open(f)]


def rr(rows, key):
    return 1 / np.array([r[key] for r in rows], dtype=float)


def diff(rows, a, b):
    d = rr(rows, a) - rr(rows, b)
    boot = np.random.default_rng(0).choice(d, (10000, len(d))).mean(axis=1)
    return f"{d.mean():+.3f} [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]"


def x1_x2():
    print("X1/X2 mechanism and hybrid (200 outbreaks each; BP damping 0.5)")
    for net in ["Hospital", "High_school", "Primary_school", "Office"]:
        rows = load(f"mechanism/diagnostics_v3_{net}_shard*.jsonl")
        if not rows:
            continue
        m = {k: rr(rows, k).mean() for k in ["Mean-field", "Belief propagation", "Mean-field, told", "Belief propagation, told",
                                              "Hybrid (MF weights, BP marginals)", "BP marginals, unweighted"]}
        grid = np.array([r["BP error (max over grid)"] for r in rows])
        told = np.array([r["BP error, told"] for r in rows])
        print(f"  {net} n={len(rows)} " + " ".join(f"{k} {v:.3f}" for k, v in m.items()))
        print(f"    told BP - told MF {diff(rows, 'Belief propagation, told', 'Mean-field, told')}"
              f" | MF - BP (end to end) {diff(rows, 'Mean-field', 'Belief propagation')}"
              f" | Hybrid - MF {diff(rows, 'Hybrid (MF weights, BP marginals)', 'Mean-field')}")
        print(f"    BP not converged (error > 1e-3): some grid point {np.mean(grid > 1e-3):.2f}, told {np.mean(told > 1e-3):.2f}")
        bad = grid > 1e-3
        if bad.any() and (~bad).any():
            print(f"    MF - BP where BP failed somewhere on the grid {diff([r for r, b in zip(rows, bad) if b], 'Mean-field', 'Belief propagation')}"
                  f" (n={bad.sum()}); where it converged everywhere {diff([r for r, b in zip(rows, bad) if not b], 'Mean-field', 'Belief propagation')} (n={(~bad).sum()})")


def x3():
    print("X3 Dawkins et al. confidence sets (target 90%) vs our conformal sets")
    conformal = json.loads((OUT / "conformal_sets.json").read_text())
    for net in ["Hospital", "Office"]:
        rows = load(f"confidence_sets/{net}_part*.jsonl")
        if not rows:
            continue
        size = np.array([r["Dawkins set size"] for r in rows])
        cover = np.mean([r["Dawkins covers"] for r in rows])
        cand = np.array([r["candidates"] for r in rows])
        sec = np.median([r["seconds"] for r in rows])
        ours = conformal.get(f"bp: {net}", {}).get("Mean-field", {})
        print(f"  {net} n={len(rows)} Dawkins: mean size {size.mean():.1f} (median {np.median(size):.0f}, of {cand.mean():.0f} candidates), "
              f"coverage {cover:.2f}, {sec:.0f} s per outbreak | ours: size {ours.get('set size for 90%')}, coverage {ours.get('test coverage')}")


def x4():
    print("X4 scalability (mean field, 1% observers)")
    for r in sorted({(x["network"], x["users"]) for x in load("scalability/*.jsonl")}):
        rows = [x for x in load("scalability/*.jsonl") if (x["network"], x["users"]) == r]
        ok = [x for x in rows if x["error"] is None]
        print(f"  {r[0]} {r[1]:>7} users, {rows[0]['contacts']:>8} contacts: n={len(rows)} "
              f"median {np.median([x['Mean-field seconds'] for x in ok]):.1f} s, peak {np.median([x['Mean-field peak MB'] for x in ok]):.0f} MB, "
              f"candidates {np.median([x['candidates'] for x in ok]):.0f}, failures {len(rows) - len(ok)}"
              + (f", MRR MF {np.mean([1 / x['Mean-field'] for x in ok]):.3f} vs distance {np.mean([1 / x['Distance center'] for x in ok]):.3f}" if ok else ""))


def x5():
    print("X5 soft-margin estimator and learned models (true rate)")
    for net in ["Hospital", "High_school", "Office", "Email-Eu"]:
        rows = [r for r in load(f"mean_field_vs_learned/{net}_0.1_*.jsonl") if r["test_rate"] == "true" and "SME (Antulov-Fantulin 2015)" in r]
        if not rows:
            continue
        keys = ["Mean-field", "SME (Antulov-Fantulin 2015)", "Distance center", "GCNSI", "Temporal GCN", "Arrival GNN"]
        print(f"  {net} n={len(rows)} " + " ".join(f"{k} {rr(rows, k).mean():.3f}" for k in keys if k in rows[0])
              + f" | MF - SME {diff(rows, 'Mean-field', 'SME (Antulov-Fantulin 2015)')}")


def x6():
    print("X6 SIR and uneven rates (BP damping 0.5)")
    for net in ["Hospital", "High_school", "Office", "Email-Eu"]:
        for scen in ["Recovery (SIR)", "Uneven rates"]:
            rows = [r for r in load(f"misspecification/{net}_seed602_shard*.jsonl") if r["scenario"] == scen]
            if not rows:
                continue
            keys = ["Mean-field", "Message passing", "Belief propagation", "Time-respecting TOA"]
            print(f"  {net} {scen} n={len(rows)} " + " ".join(f"{k} {rr(rows, k).mean():.3f}" for k in keys)
                  + f" | MF - BP {diff(rows, 'Mean-field', 'Belief propagation')} | MF - TOA {diff(rows, 'Mean-field', 'Time-respecting TOA')}")


if __name__ == "__main__":
    for f in (x1_x2, x3, x4, x5, x6):
        f()

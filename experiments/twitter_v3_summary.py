"""Twitter version 3: choose beta on the training half, then score the development and the
pre-registered test cascades (paired bootstrap intervals, Wilcoxon tests, conformal set sizes
calibrated on the development cascades and checked on the test cascades).

Run:  python -m experiments.twitter_v3_summary
"""
import glob
import json
import math

import numpy as np
from scipy.stats import wilcoxon

from experiments.benchmark_localization import PROJECT_ROOT
from experiments.real_cascades import BETAS

OUT = PROJECT_ROOT / "outputs" / "real_cascades"
BASELINES = ("Distance center", "Jordan center", "Most followers", "LPSI, tuned", "Pinto-style", "Mean-field, untrained")


def rows(which, start):
    return [json.loads(l) for f in sorted(glob.glob(str(OUT / f"twitter_v3_{which}_{start}_part*.jsonl"))) for l in open(f)]


def interval(d, rng=np.random.default_rng(0)):
    b = rng.choice(d, (10000, len(d))).mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    train = rows("train", 0)
    fit = {b: float(np.mean([r[f"log posterior at source, beta {b}"] for r in train])) for b in BETAS}
    beta = max(fit, key=fit.get)
    ours = f"Mean-field, beta {beta}"
    report = {"training cascades": len(train), "mean log posterior at source by beta": fit, "beta": beta}
    dev, test = rows("test", 0), rows("test", 300)
    for label, data in (("development (300)", dev), ("pre-registered test", test)):
        methods = {"Mean-field (v3)": [r[ours] for r in data], "Mean-field, no source prior": [r["Mean-field, beta 0.0"] for r in data]}
        methods.update({m: [r[m] for r in data] for m in BASELINES})
        block = {"cascades": len(data)}
        for m, ranks in methods.items():
            x = np.array(ranks)
            block[m] = {"MRR": float(np.mean(1 / x)), "top-1": float(np.mean(x <= 1)), "top-10": float(np.mean(x <= 10)), "median rank": float(np.median(x))}
        mine = 1 / np.array(methods["Mean-field (v3)"])
        for m in methods:
            if m != "Mean-field (v3)":
                d = mine - 1 / np.array(methods[m])
                block[m]["ours minus this (MRR)"] = float(d.mean())
                block[m]["95% interval"] = interval(d)
                block[m]["Wilcoxon p"] = float(wilcoxon(mine, 1 / np.array(methods[m])).pvalue) if np.any(d) else 1.0
        report[label] = block
    # conformal: calibrate on development, check on test
    sets = {}
    for m in ["Mean-field (v3)", "Mean-field, no source prior", *BASELINES]:
        key = ours if m == "Mean-field (v3)" else ("Mean-field, beta 0.0" if m == "Mean-field, no source prior" else m)
        cal = np.sort(np.ceil([r[key] for r in dev]))
        k = cal[math.ceil((len(cal) + 1) * 0.9) - 1]
        sets[m] = {"set size for 90%": float(k), "coverage on test": float(np.mean(np.ceil([r[key] for r in test]) <= k))}
    report["conformal (calibrated on development, checked on test)"] = sets
    (OUT / "twitter_v3_summary.json").write_text(json.dumps(report, indent=2))
    print("beta", beta, {k: round(v, 3) for k, v in fit.items()})
    for label in ("development (300)", "pre-registered test"):
        print(f"\n{label}: {report[label]['cascades']} cascades")
        for m, v in report[label].items():
            if m == "cascades":
                continue
            extra = f"  ours-this {v['ours minus this (MRR)']:+.3f} [{v['95% interval'][0]:+.3f},{v['95% interval'][1]:+.3f}] p={v['Wilcoxon p']:.1e}" if "Wilcoxon p" in v else ""
            print(f"  {m:30s} MRR {v['MRR']:.3f} top1 {v['top-1']:.2f} top10 {v['top-10']:.2f} median {v['median rank']:.0f}{extra}")
    print("\nconformal", json.dumps(sets))


if __name__ == "__main__":
    main()

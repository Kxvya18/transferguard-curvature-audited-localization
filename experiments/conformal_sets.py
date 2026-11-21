"""Source sets with guaranteed coverage (split conformal prediction on the source's rank).

For any localizer, rank everyone by its score; the nonconformity score of an outbreak is
the rank of the true source. On calibration outbreaks, take the ceil((n + 1)(1 - alpha))-th
smallest rank, k. Then for a new outbreak drawn the same way, the set of the top k people
contains the source with probability at least 1 - alpha (Vovk et al. 2005; Lei et al. 2018),
whatever the localizer and whatever the network. A better localizer gives a smaller k.

Outbreaks are split at random (fixed seed) into halves for calibration and test; the test
half checks the coverage empirically. Ties count against the method (ranks are rounded up).

Run:  python -m experiments.conformal_sets
"""

from __future__ import annotations

import glob
import json
import math

import numpy as np

from experiments.benchmark_localization import PROJECT_ROOT

ALPHA = 0.1
SOURCES = {  # label: (file pattern, row filter, methods)
    "bp": ("outputs/bp_comparison/*_0.1*.jsonl", lambda r: r.get("test_rate") == "true",
           ("Mean-field", "Belief propagation", "Message passing")),
    "twitter": ("outputs/real_cascades/twitter_test_part*.jsonl", lambda r: True,
                ("Mean-field, fitted", "Mean-field, untrained", "Distance center", "Jordan center", "LPSI, tuned", "Pinto-style")),
}


def conformal(ranks: np.ndarray, rng: np.random.Generator):
    """(k, empirical test coverage, calibration size) for one method's source ranks."""
    order = rng.permutation(len(ranks))
    calibration, test = ranks[order[: len(ranks) // 2]], ranks[order[len(ranks) // 2:]]
    n = len(calibration)
    level = math.ceil((n + 1) * (1 - ALPHA))
    k = np.inf if level > n else np.sort(np.ceil(calibration))[level - 1]
    return k, float(np.mean(np.ceil(test) <= k)), n


def main() -> None:
    out = {}
    for label, (pattern, keep, methods) in SOURCES.items():
        groups: dict[str, list[dict]] = {}
        for f in sorted(glob.glob(str(PROJECT_ROOT / pattern))):
            name = f.split("/")[-1].split("_0.1")[0] if label == "bp" else "Twitter"
            groups.setdefault(name, []).extend(r for r in map(json.loads, open(f)) if keep(r))
        for name, rows in groups.items():
            result = {}
            for m in methods:
                ranks = np.array([r[m] for r in rows if m in r], dtype=float)
                k, coverage, n = conformal(ranks, np.random.default_rng(0))
                result[m] = {"set size for 90%": k, "test coverage": round(coverage, 3), "calibration outbreaks": n}
            out[f"{label}: {name}"] = result
            print(f"{label}: {name} ({len(rows)} outbreaks)")
            for m, r in result.items():
                print(f"   {m:24s} k = {r['set size for 90%']:>6}   coverage on test half {r['test coverage']:.2f}")
    path = PROJECT_ROOT / "outputs" / "conformal_sets.json"
    path.write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()

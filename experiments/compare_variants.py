"""Compare localizer variants on identical outbreaks.

Run:  python -m experiments.compare_variants "Hospital" 0.1 40 filtered
Appends one JSON line per outbreak to outputs/variants_<tag>/.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization import mean_field as mf
from localization.spread import simulate_outbreak

VARIANTS = {
    "filtered": {"Mean-field": mf.mean_field_scores, "Filtered mean-field": mf.filtered_scores},
}


def main(name: str, fraction: float, count: int, tag: str, seed: int = 71) -> None:
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(seed)
    out = PROJECT_ROOT / "outputs" / f"variants_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name.replace(' ', '_')}_{fraction}.jsonl"
    for rate_name, rate in (("true", p), ("low", 0.4 * p), ("high", min(0.9, 1.6 * p))):
        done = 0
        while done < count:
            ob = simulate_outbreak(graph, rng, rate, horizon, fraction)
            if ob is None:
                continue
            ctx = est.build_context(graph, ob, lookback)
            row = {"dataset": name, "test_rate": rate_name}
            for label, fn in VARIANTS[tag].items():
                start = time.time()
                row[label] = rank_of_source(fn(ctx), ob.source)
                row[label + " seconds"] = time.time() - start
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            done += 1


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4])

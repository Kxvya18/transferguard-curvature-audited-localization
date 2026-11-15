"""Compare the mean-field localizer with the screen and the Boltzmann posterior on identical outbreaks.

Run:  python -m experiments.compare_mean_field "Email-Eu" 0.1 40
Appends one JSON line per outbreak to outputs/mean_field_pilot/<dataset>.jsonl.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.mean_field import mean_field_scores
from localization.spread import simulate_outbreak


def main(name: str, fraction: float, count: int, seed: int = 5) -> None:
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(seed)
    out = PROJECT_ROOT / "outputs" / "mean_field_pilot"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name.replace(' ', '_')}_{fraction}.jsonl"
    for rate_name, rate in (("true", p), ("low", 0.4 * p), ("high", min(0.9, 1.6 * p))):
        done = 0
        while done < count:
            ob = simulate_outbreak(graph, rng, rate, horizon, fraction)
            if ob is None:
                continue
            ctx = est.build_context(graph, ob, lookback)
            start = time.time()
            mean_field = rank_of_source(mean_field_scores(ctx), ob.source)
            row = {"dataset": name, "fraction": fraction, "test_rate": rate_name, "n": graph.n_nodes,
                   "Mean-field": mean_field, "seconds": time.time() - start,
                   "Screen": rank_of_source(est.temporal_screen(ctx), ob.source),
                   "Boltzmann": rank_of_source(est.boltzmann_posterior(ctx, rng).score, ob.source)}
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            done += 1


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))

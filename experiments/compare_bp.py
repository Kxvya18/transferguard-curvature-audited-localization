"""Mean field against belief propagation (SIB) and dynamic message passing, identical outbreaks.

Needs the `sib` package; run inside its environment:
    conda run -n sib-bp python -m experiments.compare_bp "Hospital" 0.1 40
Appends one JSON line per outbreak to outputs/bp_comparison/.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.belief_propagation import bp_scores
from localization.mean_field import mean_field_scores, message_passing_scores
from localization.spread import simulate_outbreak

METHODS = {"Mean-field": mean_field_scores, "Message passing": message_passing_scores, "Belief propagation": bp_scores}


def main(name: str, fraction: float, count: int, seed: int = 101) -> None:
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(seed)
    out = PROJECT_ROOT / "outputs" / "bp_comparison"
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
            for label, fn in METHODS.items():
                start = time.time()
                row[label] = rank_of_source(fn(ctx), ob.source)
                row[label + " seconds"] = time.time() - start
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            done += 1


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))

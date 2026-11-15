"""Which choices inside the mean-field localizer matter? Variants on identical outbreaks.

Run:  python -m experiments.ablate_mean_field "Email-Eu" 0.1 60
Appends one JSON line per outbreak to outputs/mean_field_ablation/.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.special import logsumexp

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.mean_field import mean_field_loglik
from localization.spread import simulate_outbreak

FINE_PROBS = tuple(np.round(np.arange(0.1, 0.95, 0.1), 2))


def variants(ctx: est.Context) -> dict[str, np.ndarray]:
    base = mean_field_loglik(ctx)
    return {
        "default": logsumexp(base, axis=0),
        "best fit instead of sum": base.max(axis=0),
        "tempered 0.5": logsumexp(0.5 * base, axis=0),
        "every start time": logsumexp(mean_field_loglik(ctx, n_start_times=64), axis=0),
        "9 probabilities": logsumexp(mean_field_loglik(ctx, prob_grid=FINE_PROBS), axis=0),
        "floor 1e-2": logsumexp(mean_field_loglik(ctx, floor=1e-2), axis=0),
        "floor 1e-6": logsumexp(mean_field_loglik(ctx, floor=1e-6), axis=0),
    }


def main(name: str, fraction: float, count: int, seed: int = 31) -> None:
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(seed)
    out = PROJECT_ROOT / "outputs" / "mean_field_ablation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name.replace(' ', '_')}_{fraction}.jsonl"
    for rate_name, rate in (("true", p), ("low", 0.4 * p)):
        done = 0
        while done < count:
            ob = simulate_outbreak(graph, rng, rate, horizon, fraction)
            if ob is None:
                continue
            ctx = est.build_context(graph, ob, lookback)
            row = {"dataset": name, "test_rate": rate_name}
            for label, score in variants(ctx).items():
                score = np.where(np.isfinite(score), score, est.INFEASIBLE)
                row[label] = rank_of_source(score, ob.source)
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            done += 1


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))

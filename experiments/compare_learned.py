"""Mean-field localizer against learned localizers on identical outbreaks.

The learned models are trained on 500 outbreaks simulated on the SAME network at the
TRUE transmission rate (their most favourable setting). The mean-field localizer gets
no training and is not told the rate. All are then tested at the true rate and at a
lower and a higher rate.

Run:  python -m experiments.compare_learned "Email-Eu" 0.1 201 60
Appends one JSON line per test outbreak to outputs/mean_field_vs_learned/.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from experiments.benchmark_localization import DATASETS, LEARNED, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.learned import predict, train_model
from localization.mean_field import mean_field_scores
from localization.spread import simulate_outbreak


def main(name: str, fraction: float, seed: int, n_test: int, n_train: int = 500) -> None:
    torch.set_num_threads(1)
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(seed)

    def contexts(count: int, rate: float) -> list[est.Context]:
        out = []
        while len(out) < count:
            ob = simulate_outbreak(graph, rng, rate, horizon, fraction)
            if ob is not None:
                out.append(est.build_context(graph, ob, lookback))
        return out

    models = {m: train_model(m, contexts(n_train, p), seed=seed, epochs=15) for m in LEARNED}
    out = PROJECT_ROOT / "outputs" / "mean_field_vs_learned"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name.replace(' ', '_')}_{fraction}_{seed}.jsonl"
    row_ids = []
    rates = os.environ.get("RATES", "true,low,high").split(",")
    for rate_name, rate in (("true", p), ("low", 0.4 * p), ("high", min(0.9, 1.6 * p))):
        if rate_name not in rates:
            continue
        for ctx in contexts(n_test, rate):
            source = ctx.outbreak.source
            row = {"dataset": name, "fraction": fraction, "seed": seed, "test_rate": rate_name,
                   "Mean-field": rank_of_source(mean_field_scores(ctx), source),
                   "Distance center": rank_of_source(est.distance_center(ctx), source),
                   "SME (Antulov-Fantulin 2015)": rank_of_source(
                       est.sme_posterior(ctx, np.random.default_rng([seed, len(row_ids)]), est.pinto_style(ctx)).score, source)}
            row_ids.append(1)
            for m, model in models.items():
                row[m] = rank_of_source(predict(m, model, ctx), source)
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

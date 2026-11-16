"""Coverage and size of 90% source sets: raw mean-field, temperature-scaled, conformal.

Calibration uses outbreaks simulated on the same network with the transmission
probability drawn from Uniform(0.1, 0.9) (the prior the localizer sums over), so
nothing about the test rate is used. Test outbreaks are at the true, a lower and a
higher rate. The simulation posterior's own 90% credible set is the reference.

Run:  python -m experiments.calibrated_sets "Hospital" 0.1 200 60
"""

from __future__ import annotations

import json
import sys

import numpy as np

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, credible_set
from localization import calibration as cal
from localization import estimators as est
from localization.mean_field import mean_field_scores
from localization.spread import simulate_outbreak


def main(name: str, fraction: float, n_cal: int, n_test: int, seed: int = 91, alpha: float = 0.1) -> None:
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(seed)

    def draw(rate):
        while True:
            ob = simulate_outbreak(graph, rng, rate, horizon, fraction)
            if ob is not None:
                return est.build_context(graph, ob, lookback)

    cal_scores, cal_sources = [], []
    for _ in range(n_cal):
        ctx = draw(float(rng.uniform(0.1, 0.9)))
        cal_scores.append(mean_field_scores(ctx))
        cal_sources.append(ctx.outbreak.source)
    temperature = cal.fit_temperature(cal_scores, cal_sources)
    threshold = cal.conformal_threshold([cal.posterior(s, temperature) for s in cal_scores], cal_sources, alpha)

    out = PROJECT_ROOT / "outputs" / "calibrated_sets"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name.replace(' ', '_')}_{fraction}.jsonl"
    for rate_name, rate in (("true", p), ("low", 0.4 * p), ("high", min(0.9, 1.6 * p))):
        for _ in range(n_test):
            ctx = draw(rate)
            src = ctx.outbreak.source
            score = mean_field_scores(ctx)
            raw = cal.posterior(score, 1.0)
            tempered = cal.posterior(score, temperature)
            conformal = cal.prediction_set(tempered, threshold)
            row = {"dataset": name, "test_rate": rate_name, "temperature": temperature, "threshold": threshold}
            for label, prob in (("raw", raw), ("tempered", tempered)):
                size, covered = credible_set(prob, src, 1 - alpha)
                row[f"{label} size"], row[f"{label} covered"] = size, covered
            row["conformal size"], row["conformal covered"] = len(conformal), bool(src in conformal)
            sim = est.boltzmann_posterior(ctx, rng)
            size, covered = credible_set(sim.probability, src, 1 - alpha)
            row["simulation size"], row["simulation covered"] = size, covered
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

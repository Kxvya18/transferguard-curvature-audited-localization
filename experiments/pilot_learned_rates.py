"""Pilot #2 (docs/pilot_new_components.md): learn each person's infectiousness from past outbreaks.

Every person has a fixed infectiousness factor (log-normal, sigma 1, mean 1), the same in every
outbreak. 150 training outbreaks (reports only; sources never used) and 60 test outbreaks.
On the test outbreaks: MF grid (the current method), MF learned, MF oracle (true factors).

Run:  python -m experiments.pilot_learned_rates <network> [train] [test]
"""
import json
import sys
import time

import numpy as np
import torch

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization import learned_rates as lr
from localization.mean_field import mean_field_scores
from localization.spread import Outbreak


def simulate(graph, rng, p, horizon, factor):
    """SI outbreak where person u transmits per contact with probability min(1, p * factor[u])."""
    n = graph.n_nodes
    t_start = int(rng.integers(0, graph.n_windows - horizon))
    active = np.unique(graph.windows[t_start].ravel())
    if not len(active):
        return None
    source = int(rng.choice(active))
    t_observe = t_start + horizon
    infected = np.full(n, np.inf)
    infected[source] = t_start
    chance = np.minimum(1.0, p * factor)
    for t in range(t_start, t_observe):
        contacts = graph.windows[t]
        for u, v in ((contacts[:, 0], contacts[:, 1]), (contacts[:, 1], contacts[:, 0])):
            can = (infected[u] <= t) & (infected[v] > t + 1)
            hit = v[can & (rng.random(len(u)) < chance[u])]
            infected[hit] = np.minimum(infected[hit], t + 1)
    observers = rng.choice(n, size=max(int(round(0.1 * n)), 3), replace=False)
    times = np.minimum(infected[observers], t_observe)
    if np.sum(times < t_observe) < 3:
        return None
    return Outbreak(source, t_start, t_observe, observers, times)


def outbreaks(graph, rng, p, horizon, lookback, factor, count):
    out = []
    while len(out) < count:
        ob = simulate(graph, rng, p, horizon, factor)
        if ob is not None:
            out.append(est.build_context(graph, ob, lookback))
    return out


def main(name, n_train=150, n_test=60):
    torch.set_num_threads(4)
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(1301)
    rng = np.random.default_rng(1301)
    factor = rng.lognormal(-0.5, 1.0, graph.n_nodes)
    train = outbreaks(graph, rng, p, horizon, lookback, factor, n_train)
    test = outbreaks(graph, rng, p, horizon, lookback, factor, n_test)
    folder = PROJECT_ROOT / "outputs" / "pilot_learned_rates"
    folder.mkdir(parents=True, exist_ok=True)
    tag = name.replace(" ", "_")
    start = time.time()
    preps = [lr.Prepared(c) for c in train]
    learned = lr.fit(preps, graph.n_nodes, p, log=lambda s, v, base: print(f"{name} step {s} evidence {v:.3f} base {base:.3f}", flush=True))
    fit_seconds = time.time() - start
    oracle = torch.tensor(np.minimum(1.0, p * factor), dtype=torch.float32)
    corr = float(np.corrcoef(np.log(learned.numpy()), np.log(oracle.numpy()))[0, 1])
    with (folder / f"{tag}_fit.json").open("w") as h:
        json.dump({"network": name, "train": n_train, "fit seconds": fit_seconds,
                   "corr log learned vs log true": corr}, h)
    with (folder / f"{tag}.jsonl").open("w") as h:
        for i, ctx in enumerate(test):
            prep = lr.Prepared(ctx)
            s = ctx.outbreak.source
            row = {"network": name, "outbreak": i,
                   "MF grid": rank_of_source(mean_field_scores(ctx), s),
                   "MF learned": rank_of_source(lr.scores(prep, learned), s),
                   "MF oracle": rank_of_source(lr.scores(prep, oracle), s)}
            h.write(json.dumps(row) + "\n")
    print(name, "fit", round(fit_seconds), "s, corr", round(corr, 3))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:4]))

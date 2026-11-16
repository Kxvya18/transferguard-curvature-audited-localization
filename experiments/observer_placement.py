"""Which person to test next: information gain against simple testing strategies.

Each outbreak starts with a few random observers (5% of people, at least 3
infected). Then each strategy picks people to test one at a time, up to a budget;
a test reveals that person's true infection time, or that they were healthy at
t_observe. After every test the source is ranked by the same mean-field
localizer, so only the choice of whom to test differs between strategies.

Run:  python -m experiments.observer_placement "Hospital" 30 [budget] [seed]
Appends one JSON line per outbreak and strategy to outputs/observer_placement/.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.mean_field import mean_field_scores
from localization.placement import information_gain
from localization.spread import Outbreak, spread

FRACTION = 0.05


def outbreak_with_truth(graph, rng, p, horizon):
    """Like spread.simulate_outbreak, but also returns everyone's infection time."""
    n = graph.n_nodes
    n_obs = max(int(round(FRACTION * n)), 3)
    while True:
        t_start = int(rng.integers(0, graph.n_windows - horizon))
        active = np.unique(graph.windows[t_start].ravel())
        if not len(active):
            continue
        source = int(rng.choice(active))
        start = np.full((1, n), np.inf)
        start[0, source] = t_start
        t_observe = t_start + horizon
        times = np.minimum(spread(graph, start, t_start, t_observe, p, rng)[0], t_observe)
        observers = rng.choice(n, size=n_obs, replace=False)
        if np.sum(times[observers] < t_observe) >= 3:
            return Outbreak(source, t_start, t_observe, observers, times[observers]), times


def pick(strategy: str, ctx: est.Context, rng: np.random.Generator) -> int:
    ob = ctx.outbreak
    untested = np.setdiff1d(np.arange(ctx.graph.n_nodes), ob.observers)
    if strategy == "Random":
        return int(rng.choice(untested))
    if strategy == "Highest degree":
        degree = np.array([ctx.static.degree(v) if v in ctx.static else 0 for v in untested])
        return int(untested[np.argmax(degree)])
    if strategy == "Contact tracing":
        # Most contacts with people already known to be infected; ties go to higher degree.
        infected = set(ob.infected_observers.tolist())
        def key(v):
            if v not in ctx.static:
                return (0, 0)
            nbrs = set(ctx.static.neighbors(v))
            return (len(nbrs & infected), len(nbrs))
        return int(max(untested, key=key))
    if strategy == "Test top suspect":
        score = mean_field_scores(ctx)
        score[ob.observers] = -np.inf
        return int(np.argmax(score))
    if strategy == "Information gain (ours)":
        return int(np.argmax(np.where(np.isin(np.arange(ctx.graph.n_nodes), untested), information_gain(ctx), -1)))
    raise ValueError(strategy)


STRATEGIES = ["Random", "Highest degree", "Contact tracing", "Test top suspect", "Information gain (ours)"]


def main(name: str, count: int, budget: int = 10, seed: int = 301) -> None:
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(seed)
    out = PROJECT_ROOT / "outputs" / "observer_placement"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name.replace(' ', '_')}.jsonl"
    done = sum(1 for _ in path.open()) // len(STRATEGIES) if path.exists() else 0
    for k in range(count):
        first, truth = outbreak_with_truth(graph, rng, p, horizon)
        if k < done:  # resume: same outbreaks, already scored
            continue
        for strategy in STRATEGIES:
            pick_rng = np.random.default_rng([seed, k])
            ob = first
            ranks, seconds = [], 0.0
            for step in range(budget + 1):
                ctx = est.build_context(graph, ob, lookback)
                ranks.append(rank_of_source(mean_field_scores(ctx), ob.source))
                if step == budget:
                    break
                start = time.time()
                v = pick(strategy, ctx, pick_rng)
                seconds += time.time() - start
                ob = Outbreak(ob.source, ob.t_start, ob.t_observe, np.r_[ob.observers, v], np.r_[ob.observed_times, truth[v]])
            row = {"dataset": name, "strategy": strategy, "ranks": ranks, "seconds per pick": seconds / budget}
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), *(int(a) for a in sys.argv[3:5]))

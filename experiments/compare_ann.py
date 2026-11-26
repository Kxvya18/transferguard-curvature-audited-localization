"""Mean field against the autoregressive-network localizer of Biazzo et al. (Sci. Rep. 2022).

The network method is told the true start and rate (it needs them); mean field is reported both
told and not told. Rules fixed before running: docs/ieee_trans_review.md section 11.

Run (GitHub Actions; needs torch and annfore):
  python -m experiments.compare_ann validate <count> <beta steps> <samples> [seed]
      tiny temporal graphs: exact source posterior (enumerating every transmission coin) vs the
      network and vs mean field told; checks our conversion to annfore and how close it gets.
  python -m experiments.compare_ann run <network> <count> <beta steps> <samples> [seed]
      outbreaks generated exactly as experiments.compare_bp (same seed -> same outbreaks, so rows
      pair with the belief-propagation tables). SHARD="i/n" splits outbreaks over processes.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
from scipy.special import logsumexp

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.ann_biazzo import ann_source_probs
from localization.mean_field import mean_field_loglik, mean_field_scores
from localization.spread import Outbreak, simulate_outbreak, spread
from localization.temporal_graph import TemporalGraph, _join_times

OUT = PROJECT_ROOT / "outputs" / "ann_comparison"


def told_mean_field(ctx, ob, p):
    """Mean-field log-likelihood per node with the true start and rate."""
    return mean_field_loglik(replace(ctx, t_lo=ob.t_start), prob_grid=(p,), n_start_times=1)[0]


def normalized(loglik):
    out = np.zeros(len(loglik))
    ok = np.isfinite(loglik)
    if ok.any():
        out[ok] = np.exp(loglik[ok] - logsumexp(loglik[ok]))
    return out


def exact_posterior(graph, ob, p):
    """P(s | reports) for SI started at ob.t_start with rate p, uniform prior over nodes, by
    enumerating every transmission coin (one coin per contact and direction, as `spread`)."""
    contacts = [(t, int(u), int(v)) for t in range(ob.t_start, ob.t_observe) for a, b in graph.windows[t]
                for u, v in ((a, b), (b, a))]
    like = np.zeros(graph.n_nodes)
    for s in range(graph.n_nodes):
        for coins in itertools.product([0, 1], repeat=len(contacts)):
            times = np.full(graph.n_nodes, np.inf)
            times[s] = ob.t_start
            k = 0
            for t in range(ob.t_start, ob.t_observe):
                now = times.copy()
                while k < len(contacts) and contacts[k][0] == t:
                    _, u, v = contacts[k]
                    if coins[k] and now[u] <= t and times[v] > t + 1:
                        times[v] = t + 1
                    k += 1
            seen = np.minimum(times[ob.observers], ob.t_observe)
            if np.array_equal(seen, ob.observed_times):
                like[s] += np.prod([p if c else 1 - p for c in coins])
    return like / like.sum()


def tiny_outbreak(rng):
    while True:
        n, T = int(rng.integers(5, 8)), int(rng.integers(4, 6))
        windows = []
        for _ in range(T):
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < 0.3]
            windows.append(np.array(pairs, dtype=np.int64).reshape(-1, 2))
        if not (5 <= sum(len(w) for w in windows) <= 7) or not len(windows[0]):
            continue  # <= 7 contacts x 2 directions = 14 coins: exact enumeration stays fast
        graph = TemporalGraph("tiny", n, windows, _join_times(n, windows))
        p = float(rng.choice([0.3, 0.6]))
        source = int(rng.choice(np.unique(windows[0].ravel())))
        start = np.full((1, n), np.inf)
        start[0, source] = 0
        times = spread(graph, start, 0, T, p, rng)[0]
        observers = np.sort(rng.choice(n, size=n // 2 + 1, replace=False))
        observed = np.minimum(times[observers], T)
        if np.sum(observed < T) >= 1:
            return graph, Outbreak(source, 0, T, observers, observed.astype(float)), p


def validate(count, steps, samples, seed=2024):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    path = OUT / (f"validate_{steps}x{samples}.jsonl" if seed == 2024 else f"validate_{steps}x{samples}_seed{seed}.jsonl")
    for k in range(count):
        graph, ob, p = tiny_outbreak(rng)
        exact = exact_posterior(graph, ob, p)
        ann, seconds = ann_source_probs(graph, ob, 0, p, steps, samples, seed=k)
        ctx = est.build_context(graph, ob, ob.t_observe)
        mf = normalized(told_mean_field(ctx, ob, p))
        row = {"n": graph.n_nodes, "p": p, "TV exact-ANN": 0.5 * np.abs(exact - ann).sum(),
               "TV exact-MF told": 0.5 * np.abs(exact - mf).sum(), "ANN seconds": seconds,
               "exact": exact.round(4).tolist(), "ANN": ann.round(4).tolist(), "MF told": mf.round(4).tolist()}
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(k, f"TV ANN {row['TV exact-ANN']:.3f} MF {row['TV exact-MF told']:.3f} {seconds:.0f}s", flush=True)


def run(name, count, steps, samples, seed=101):
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(seed)
    shard, shards = map(int, os.environ.get("SHARD", "0/1").split("/"))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name.replace(' ', '_')}_{steps}x{samples}_seed{seed}_shard{shard}" + (f"of{shards}" if shards != 4 else "") + ".jsonl"
    done_rows = sum(1 for _ in path.open()) if path.exists() else 0
    done = mine = 0
    while done < count:
        ob = simulate_outbreak(graph, rng, p, horizon, 0.1)  # as compare_bp at fraction 0.1, rate "true"
        if ob is None:
            continue
        if done % shards != shard:
            done += 1
            continue
        mine += 1
        if mine <= done_rows:
            done += 1
            continue
        ctx = est.build_context(graph, ob, lookback)
        row = {"dataset": name, "outbreak": done, "beta steps": steps, "samples": samples}
        start = time.time()
        row["Mean-field"] = rank_of_source(mean_field_scores(ctx), ob.source)
        row["Mean-field seconds"] = time.time() - start
        start = time.time()
        told = told_mean_field(ctx, ob, p)
        row["Mean-field, told"] = rank_of_source(np.where(np.isfinite(told), told, est.INFEASIBLE), ob.source)
        row["Mean-field, told seconds"] = time.time() - start
        ann, seconds = ann_source_probs(graph, ob, ob.t_start, p, steps, samples, seed=done)
        row["ANN (Biazzo 2022), told"] = rank_of_source(np.where(ann > 0, np.log(np.maximum(ann, 1e-300)), est.INFEASIBLE), ob.source)
        row["ANN seconds"] = seconds
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(done, {k: v for k, v in row.items() if "seconds" not in k}, f"{seconds:.0f}s", flush=True)
        done += 1


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), *(int(a) for a in sys.argv[5:6]))
    else:
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), *(int(a) for a in sys.argv[6:7]))

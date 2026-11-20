"""Real outbreaks: retweet cascades on the Twitter follower network (Higgs boson dataset).

Data (SNAP, De Domenico et al. 2013), placed in data/raw/higgs/:
    higgs-social_network.edgelist.gz   who follows whom (456,626 users)
    higgs-activity_time.txt.gz         retweets with timestamps, "A B time RT" = A retweeted B

Each user B whose posts were retweeted by at least 10 people is one real outbreak:
B is the true source, and a retweeter is "infected" at the time of their first retweet
of B. Only the first 50 retweeters are used. The network is the follower graph
(undirected) on the retweeters and everyone they follow or are followed by; the source
is not added on purpose, so an outbreak is kept only if the source is among those
neighbours (otherwise no network method could find it). Follower links are contacts in
every window. The first 50 retweets are spread over windows 4 .. 19 (window length
= their time span / 16), observation is at window 20, and the source's own posting
time is unknown, as in the simulated benchmark. A random 10% of people are tested
(never the source): retweeters report their window, everyone else reports healthy.

Run:  python -m experiments.real_cascades 300 [fraction] [seed]
"""

from __future__ import annotations

import gzip
import json
import sys
import time

import numpy as np
import scipy.sparse as sp

from experiments.benchmark_localization import PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.mean_field import mean_field_scores, message_passing_scores
from localization.spread import Outbreak
from localization.temporal_graph import TemporalGraph

RAW = PROJECT_ROOT / "data" / "raw" / "higgs"
WINDOWS, FIRST, MAX_INFECTED, MIN_INFECTED, MAX_NODES = 20, 4, 50, 10, 4000
METHODS = {"Mean-field": mean_field_scores, "Message passing": message_passing_scores,
           "Time-respecting TOA": est.temporal_screen, "Jordan center": est.jordan_center,
           "Distance center": est.distance_center}


def load():
    edges = np.loadtxt(gzip.open(RAW / "higgs-social_network.edgelist.gz"), dtype=np.int64)
    n = int(edges.max()) + 1
    adjacency = sp.coo_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(n, n))
    adjacency = ((adjacency + adjacency.T) > 0).tocsr()
    cascades: dict[int, dict[int, int]] = {}
    for line in gzip.open(RAW / "higgs-activity_time.txt.gz", "rt"):
        a, b, t, kind = line.split()
        if kind == "RT" and a != b:
            first = cascades.setdefault(int(b), {})
            first[int(a)] = min(first.get(int(a), int(t)), int(t))
    return adjacency, {s: c for s, c in cascades.items() if len(c) >= MIN_INFECTED}


def outbreak_graph(adjacency, source: int, retweets: dict[int, int]):
    """(graph, local source, local infected nodes, their windows) or None if the source is unreachable."""
    order = sorted(retweets, key=retweets.get)[:MAX_INFECTED]
    times = np.array([retweets[u] for u in order], dtype=float)
    infected = np.array(order)
    nodes = np.unique(np.r_[infected, adjacency[infected].indices])
    if source not in set(nodes.tolist()) or len(nodes) > MAX_NODES:
        return None
    local = {int(v): i for i, v in enumerate(nodes)}
    sub = sp.triu(adjacency[nodes][:, nodes], k=1).tocoo()
    pairs = np.c_[sub.row, sub.col].astype(np.int64)
    windows = [pairs] * WINDOWS
    span = max(times[-1] - times[0], 1.0)
    window_of = FIRST + np.minimum(((times - times[0]) / span * 16).astype(int), 15)
    graph = TemporalGraph(f"Twitter cascade {source}", len(nodes), windows, np.zeros(len(nodes), dtype=int))
    return graph, local[source], np.array([local[int(u)] for u in infected]), window_of


def main(count: int, fraction: float = 0.1, seed: int = 1001) -> None:
    adjacency, cascades = load()
    rng = np.random.default_rng(seed)
    sources = rng.permutation(sorted(cascades))
    out = PROJECT_ROOT / "outputs" / "real_cascades"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"twitter_{int(round(fraction * 100))}pct.jsonl"
    done = sum(1 for _ in path.open()) if path.exists() else 0
    kept = skipped = 0
    for source in sources:
        if kept >= count:
            break
        built = outbreak_graph(adjacency, int(source), cascades[source])
        observer_rng = np.random.default_rng([seed, int(source)])
        if built is None:
            skipped += 1
            continue
        graph, src, infected, window_of = built
        status = np.full(graph.n_nodes, WINDOWS, dtype=float)
        status[infected] = window_of
        pool = np.setdiff1d(np.arange(graph.n_nodes), [src])
        observers = observer_rng.choice(pool, size=max(int(round(fraction * graph.n_nodes)), 3), replace=False)
        if np.sum(status[observers] < WINDOWS) < 3:
            skipped += 1
            continue
        kept += 1
        if kept <= done:
            continue
        ob = Outbreak(src, 0, WINDOWS, observers, status[observers])
        ctx = est.build_context(graph, ob, WINDOWS)
        row = {"source": int(source), "people": graph.n_nodes, "retweeters": len(infected),
               "infected observers": int(np.sum(status[observers] < WINDOWS)), "skipped so far": skipped}
        for label, fn in METHODS.items():
            start = time.time()
            row[label] = rank_of_source(fn(ctx), src)
            row[label + " seconds"] = time.time() - start
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]), *(float(a) for a in sys.argv[2:3]), *(int(a) for a in sys.argv[3:4]))

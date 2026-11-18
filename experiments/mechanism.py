"""Why does mean field beat belief propagation on real contact networks?

Two candidate causes, separated on synthetic contact sequences where each can be
set on its own:

- loops: the network of who-meets-whom has cycles (BP is exact only without them);
- repeats: the same pair meets again and again within the outbreak window.

A backbone graph on 100 people is drawn (a random tree, or a random graph with
mean degree 4 or 8). Each window, contacts are drawn uniformly from the backbone
edges; the number per window sets how often each pair meets on average over the
lookback period (1, 3 or 9 times). Mean field, message passing and belief
propagation rank the source of the same outbreaks in every cell.

The script also measures both quantities on the real networks, so the real gaps
(outputs/bp_comparison) can be placed on the same axes.

Needs `sib`:
    conda run -n sib-bp python -m experiments.mechanism synthetic tree 3 30
    conda run -n sib-bp python -m experiments.mechanism real
    conda run -n sib-bp python -m experiments.mechanism reference "Hospital" "No repeats" 30
"""

from __future__ import annotations

import json
import sys
import time

import networkx as nx
import numpy as np

from experiments.benchmark_localization import DATASETS, PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.belief_propagation import bp_scores
from localization.mean_field import mean_field_scores, message_passing_scores
from localization.spread import simulate_outbreak
from localization.temporal_graph import TemporalGraph, _join_times

# Per-contact probability falls as meetings repeat, so each pair has a similar chance of
# passing the infection over the lookback period and outbreaks stay comparable in size.
N, WINDOWS, HORIZON, LOOKBACK, FRACTION = 100, 150, 20, 40, 0.2
RATE = {1: 0.9, 3: 0.5, 9: 0.25}
BACKBONES = {"tree": None, "degree 4": 4, "degree 8": 8}
METHODS = {"Mean-field": mean_field_scores, "Message passing": message_passing_scores, "Belief propagation": bp_scores}


def backbone(kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "tree":
        graph = nx.random_labeled_tree(N, seed=int(rng.integers(1 << 31))) if hasattr(nx, "random_labeled_tree") \
            else nx.random_tree(N, seed=int(rng.integers(1 << 31)))
    else:
        graph = nx.gnm_random_graph(N, N * BACKBONES[kind] // 2, seed=int(rng.integers(1 << 31)))
    return np.array(graph.edges())


def contact_sequence(kind: str, repeats: int, rng: np.random.Generator) -> TemporalGraph:
    """Each backbone edge is used `repeats` times on average per LOOKBACK windows."""
    edges = backbone(kind, rng)
    per_window = max(1, round(repeats * len(edges) / LOOKBACK))
    windows = [np.unique(np.sort(edges[rng.integers(len(edges), size=per_window)], axis=1), axis=0)
               for _ in range(WINDOWS)]
    return TemporalGraph(f"{kind}, {repeats}x", N, windows, _join_times(N, windows))


def structure(graph: TemporalGraph, rng: np.random.Generator, samples: int = 50) -> dict:
    """Averaged over random lookback periods: meetings per pair, and cycle rank per node
    of the who-met-whom graph (edges - nodes + components; 0 for a forest)."""
    repeat, loops = [], []
    for _ in range(samples):
        t0 = int(rng.integers(0, graph.n_windows - LOOKBACK_OF[graph.name]))
        contacts = np.concatenate([w for w in graph.windows[t0:t0 + LOOKBACK_OF[graph.name]] if len(w)] or [np.zeros((0, 2), int)])
        if not len(contacts):
            continue
        pairs = np.unique(np.sort(contacts, axis=1), axis=0)
        g = nx.Graph(pairs.tolist())
        repeat.append(len(contacts) / len(pairs))
        loops.append((g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)) / g.number_of_nodes())
    return {"meetings per pair": float(np.mean(repeat)), "cycle rank per node": float(np.mean(loops))}


LOOKBACK_OF: dict[str, int] = {}


def synthetic(kind: str, repeats: int, count: int, seed: int = 701) -> None:
    rng = np.random.default_rng([seed, repeats, list(BACKBONES).index(kind)])
    out = PROJECT_ROOT / "outputs" / "mechanism"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "synthetic.jsonl"
    done = sum(1 for line in path.open() if json.loads(line)["cell"] == f"{kind}, {repeats}x") if path.exists() else 0
    graph = contact_sequence(kind, repeats, rng)
    LOOKBACK_OF[graph.name] = LOOKBACK
    shape = structure(graph, rng)
    k = 0
    while k < count:
        ob = simulate_outbreak(graph, rng, RATE[repeats], HORIZON, FRACTION, max_tries=2000)
        if ob is None:
            continue
        k += 1
        if k <= done:
            continue
        ctx = est.build_context(graph, ob, LOOKBACK)
        row = {"cell": graph.name, "backbone": kind, "repeats": repeats, **shape,
               "infected observers": int(np.sum(ob.observed_times < ob.t_observe))}
        for label, fn in METHODS.items():
            start = time.time()
            row[label] = rank_of_source(fn(ctx), ob.source)
            row[label + " seconds"] = time.time() - start
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


def randomized(graph: TemporalGraph, kind: str, rng: np.random.Generator) -> TemporalGraph:
    """Reference models (Gauvin et al., 2022) that remove one property of a real contact log.

    - "No repeats": each window keeps its active people and its number of contacts, but
      the contacts are redrawn as random pairs among those people, so pairs rarely meet twice.
    - "Shuffled times": every contact keeps its pair but gets the window of a randomly
      chosen other contact, so who-met-whom and how often are kept, timing is not.
    """
    if kind == "No repeats":
        windows = []
        for w in graph.windows:
            people = np.unique(w.ravel())
            if len(people) < 2:
                windows.append(w)
                continue
            pairs = np.array([rng.choice(people, 2, replace=False) for _ in range(len(w))])
            windows.append(np.unique(np.sort(pairs, axis=1), axis=0))
    else:
        stamp = np.concatenate([np.full(len(w), t) for t, w in enumerate(graph.windows)])
        pairs = np.concatenate([w for w in graph.windows if len(w)])
        stamp = rng.permutation(stamp)
        windows = [np.unique(pairs[stamp == t], axis=0) if np.any(stamp == t) else np.zeros((0, 2), np.int64)
                   for t in range(graph.n_windows)]
    return TemporalGraph(f"{graph.name}, {kind}", graph.n_nodes, windows, _join_times(graph.n_nodes, windows))


def reference(name: str, kind: str, count: int, seed: int = 801) -> None:
    loader, p, horizon, lookback = DATASETS[name]
    rng = np.random.default_rng(seed)
    graph = loader(seed)
    if kind != "Original":
        graph = randomized(graph, kind, rng)
    LOOKBACK_OF[graph.name] = lookback
    shape = structure(graph, rng)
    out = PROJECT_ROOT / "outputs" / "mechanism"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "reference_models.jsonl"
    done = sum(1 for line in path.open() if json.loads(line)["model"] == kind and json.loads(line)["dataset"] == name) \
        if path.exists() else 0
    k = 0
    while k < count:
        ob = simulate_outbreak(graph, rng, p, horizon, 0.1)
        if ob is None:
            continue
        k += 1
        if k <= done:
            continue
        ctx = est.build_context(graph, ob, lookback)
        row = {"dataset": name, "model": kind, **shape}
        for label, fn in METHODS.items():
            start = time.time()
            row[label] = rank_of_source(fn(ctx), ob.source)
            row[label + " seconds"] = time.time() - start
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


def real() -> None:
    out = PROJECT_ROOT / "outputs" / "mechanism"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (loader, p, horizon, lookback) in DATASETS.items():
        graph = loader(101)
        LOOKBACK_OF[graph.name] = lookback
        rows.append({"dataset": name, **structure(graph, np.random.default_rng(0))})
        print(rows[-1], flush=True)
    (out / "real_structure.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "real":
        real()
    elif sys.argv[1] == "reference":
        reference(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        synthetic(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

"""Real outbreaks: retweet cascades on the Twitter follower network (Higgs boson dataset).

Data (SNAP, De Domenico et al. 2013), in data/raw/higgs/ (or $HIGGS_DIR):
    higgs-social_network.edgelist.gz   "a b" = user a follows user b
    higgs-activity_time.txt.gz         "a b time RT" = a retweeted b

Each user B retweeted by at least 10 people is one real outbreak: B is the true source,
and a retweeter is infected at their first retweet of B. The first 50 retweeters are
used. The network is the follower graph on the retweeters and everyone they follow or
are followed by; a post passes from the followed user to the follower (directed). The
source is not added on purpose, so an outbreak is kept only if the source is among
those people. Follower links are open in every window. The first 50 retweets are spread
over windows 4 .. 19, observation is at window 20, and the source's posting time is
unknown. A random 10% of people are tested (never the source).

Protocol (no information from the test outbreaks is used to build any method):
- outbreaks are ordered by their first retweet; the earlier half is the training set,
  the later half the test set;
- "fit": the spreading model is fitted by maximum likelihood at the known sources of
  training outbreaks: rate for everyone, a separate rate for the source, the chance a
  spreader stops spreading each window, and the fraction of people who ever retweet;
  every baseline with a free parameter (LPSI's alpha) is tuned on the same outbreaks,
  by mean reciprocal rank;
- "test": every method ranks every person in each test outbreak.

Run:
    python -m experiments.real_cascades fit 150
    python -m experiments.real_cascades test 300 [shard] [shards]
"""

from __future__ import annotations

import gzip
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp

from experiments.benchmark_localization import PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.mean_field import mean_field_loglik, mean_field_scores
from localization.spread import Outbreak
from localization.temporal_graph import TemporalGraph

RAW = Path(os.environ.get("HIGGS_DIR", PROJECT_ROOT / "data" / "raw" / "higgs"))
OUT = Path(os.environ.get("OUT_DIR", PROJECT_ROOT / "outputs" / "real_cascades"))
WINDOWS, FIRST, MAX_INFECTED, MIN_INFECTED, MAX_NODES = 20, 4, 50, 10, 4000
FRACTION, SEED = 0.1, 1001

# Model grid searched when fitting: (rate, source rate, stop-spreading chance, fraction who
# can retweet, chance per window of a retweet from outside the follower graph)
FIT_GRID = list(itertools.product((1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2), (0.003, 0.01, 0.03, 0.1, 0.3),
                                  (0.0, 0.5, 1.0), (1.0, 0.3, 0.1), (1e-4, 1e-3, 1e-2)))
# Second stage, around the first stage's best model (whose rate was the smallest tried):
# reaches down to no spread between non-sources at all. Selected with FIT_STAGE=2.
if os.environ.get("FIT_STAGE") == "2":
    FIT_GRID = list(itertools.product((0.0, 1e-5, 3e-5, 1e-4, 3e-4), (0.003, 0.006, 0.01, 0.02), (0.0,), (1.0, 0.3),
                                      (3e-4, 1e-3, 3e-3)))
# Third stage: each outbreak's source has its own broadcast rate (a big account reaches a
# smaller share of its followers than a small one); its prior over SOURCE_RATES is fitted
# by EM on the training outbreaks, for each setting of the other parameters. FIT_STAGE=3.
SOURCE_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3)
if os.environ.get("FIT_STAGE") == "3":
    FIT_GRID = [(r, s, 0.0, f, l) for r in (0.0, 1e-5, 1e-4) for f in (1.0, 0.3) for l in (3e-4, 1e-3, 3e-3) for s in SOURCE_RATES]
# Fourth stage: stage 3's best fraction and leak were at the edges of their ranges, so
# both ranges are extended past them. FIT_STAGE=4.
if os.environ.get("FIT_STAGE") == "4":
    FIT_GRID = [(0.0, s, 0.0, f, l) for f in (0.3, 0.1, 0.03) for l in (1e-3, 3e-3, 1e-2, 3e-2) for s in SOURCE_RATES]
FLOOR = 1e-12  # the leak gives every report a proper probability, so no floor is needed
LPSI_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)


def load():
    """Directed follow matrix (row a follows column b) and cascades {source: {retweeter: time}}."""
    edges = np.loadtxt(gzip.open(RAW / "higgs-social_network.edgelist.gz"), dtype=np.int64)
    n = int(edges.max()) + 1
    follows = sp.csr_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(n, n))
    cascades: dict[int, dict[int, int]] = {}
    for line in gzip.open(RAW / "higgs-activity_time.txt.gz", "rt"):
        a, b, t, kind = line.split()
        if kind == "RT" and a != b:
            first = cascades.setdefault(int(b), {})
            first[int(a)] = min(first.get(int(a), int(t)), int(t))
    return follows, {s: c for s, c in cascades.items() if len(c) >= MIN_INFECTED}


def outbreak(follows, either, source: int, retweets: dict[int, int]):
    """Context and local source index for one cascade, or None if it is unusable."""
    order = sorted(retweets, key=retweets.get)[:MAX_INFECTED]
    times = np.array([retweets[u] for u in order], dtype=float)
    nodes = np.unique(np.r_[order, either[order].indices])
    if source not in set(nodes.tolist()) or len(nodes) > MAX_NODES:
        return None
    local = {int(v): i for i, v in enumerate(nodes)}
    sub = follows[nodes][:, nodes].tocoo()
    pairs = np.c_[sub.col, sub.row].astype(np.int64)  # followed -> follower
    graph = TemporalGraph(f"Twitter cascade {source}", len(nodes), [pairs] * WINDOWS, np.zeros(len(nodes), dtype=int))
    span = max(times[-1] - times[0], 1.0)
    status = np.full(len(nodes), WINDOWS, dtype=float)
    status[[local[int(u)] for u in order]] = FIRST + np.minimum(((times - times[0]) / span * 16).astype(int), 15)
    src = local[source]
    pool = np.setdiff1d(np.arange(len(nodes)), [src])
    observers = np.random.default_rng([SEED, source]).choice(pool, size=max(int(round(FRACTION * len(nodes))), 3), replace=False)
    if np.sum(status[observers] < WINDOWS) < 3:
        return None
    return est.build_context(graph, Outbreak(src, 0, WINDOWS, observers, status[observers]), WINDOWS), src


def split(cascades):
    """Sources ordered by first retweet: earlier half for training, later half for testing."""
    order = sorted(cascades, key=lambda s: min(cascades[s].values()))
    half = len(order) // 2
    return order[:half], order[half:]


def model_kwargs(rate, source_rate, stop, fraction, leak):
    return dict(prob_grid=(rate,), seed_prob=source_rate, recovery_grid=(stop,), susceptible_grid=(fraction,),
                leak=leak, floor=FLOOR, directed=True)


def case_loglik(case) -> np.ndarray:
    """Log-likelihood of one training outbreak at its true source, for every model in the grid."""
    ctx, src = case
    return np.array([logsumexp(mean_field_loglik(ctx, candidates=np.array([src]), prune=False, **model_kwargs(*params))[:, src])
                     for params in FIT_GRID])  # start time summed out


def fit(count: int) -> None:
    follows, cascades = load()
    either = ((follows + follows.T) > 0).tocsr()
    train, _ = split(cascades)
    cases = []
    for source in train:
        built = outbreak(follows, either, source, cascades[source])
        if built is not None:
            cases.append(built)
        if len(cases) >= count:
            break
    with Pool(int(os.environ.get("WORKERS", 4))) as pool:
        per_case = np.array(pool.map(case_loglik, cases))
    loglik = per_case.sum(axis=0)
    best = FIT_GRID[int(np.argmax(loglik))]
    prior = None
    if os.environ.get("FIT_STAGE") in ("3", "4"):
        # For each setting of the other parameters, fit the source-rate prior by EM and keep
        # the setting with the highest marginal likelihood.
        bases = sorted({(g[0], g[2], g[3], g[4]) for g in FIT_GRID})
        scored = []
        for base in bases:
            cols = [i for i, g in enumerate(FIT_GRID) if (g[0], g[2], g[3], g[4]) == base]
            sub = per_case[:, cols]
            w = np.full(len(cols), 1.0 / len(cols))
            for _ in range(500):
                post = np.exp(np.log(w) + sub - logsumexp(np.log(w) + sub, axis=1, keepdims=True))
                w = np.maximum(post.mean(axis=0), 1e-12)
            scored.append((float(logsumexp(np.log(w) + sub, axis=1).sum()), base, w))
        marginal, base, w = max(scored, key=lambda x: x[0])
        best = (base[0], None, base[1], base[2], base[3])
        prior = {str(r): float(x) for r, x in zip(SOURCE_RATES, w)}
    alpha_mrr = {a: float(np.mean([1 / rank_of_source(est.lpsi(ctx, alpha=a), src) for ctx, src in cases])) for a in LPSI_ALPHAS}
    result = {"training outbreaks": len(cases), "model": dict(zip(("rate", "source rate", "stop", "fraction", "leak"), best)),
              "log-likelihood by model": {str(k): float(v) for k, v in zip(FIT_GRID, loglik)},
              "LPSI alpha": max(alpha_mrr, key=alpha_mrr.get), "LPSI training MRR": alpha_mrr,
              "source rate prior": prior}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "twitter_fit.json").write_text(json.dumps(result, indent=2))
    print(result["model"], result["LPSI alpha"], flush=True)


def mixture_scores(ctx, model, prior) -> np.ndarray:
    """Mean-field scores with the source's own rate summed out under its fitted prior."""
    rate, _, stop, fraction, leak = model
    terms = [np.log(w) + logsumexp(mean_field_loglik(ctx, prune=False, **model_kwargs(rate, float(s), stop, fraction, leak)), axis=0)
             for s, w in prior.items() if w > 1e-6]
    score = logsumexp(np.array(terms), axis=0)
    return np.where(np.isfinite(score), score, est.INFEASIBLE)


def test(count: int, shard: int = 0, shards: int = 1) -> None:
    fitted = json.loads((OUT / "twitter_fit.json").read_text())
    single = json.loads((OUT / "twitter_fit_stage2.json").read_text())
    model, alpha = tuple(fitted["model"].values()), fitted["LPSI alpha"]
    one_rate = tuple(single["model"].values())
    methods = {
        "Mean-field, fitted": lambda ctx: mixture_scores(ctx, model, fitted["source rate prior"]),
        "Mean-field, one source rate": lambda ctx: mean_field_scores(ctx, prune=False, **model_kwargs(*one_rate)),
        "Mean-field, untrained": lambda ctx: mean_field_scores(ctx, prob_grid=(0.003, 0.01, 0.03, 0.1, 0.3, 0.9)),
        "LPSI, tuned": lambda ctx: est.lpsi(ctx, alpha=alpha),
        "Pinto-style": est.pinto_style,
        "Jordan center": est.jordan_center,
        "Distance center": est.distance_center,
    }
    follows, cascades = load()
    either = ((follows + follows.T) > 0).tocsr()
    _, sources = split(cascades)
    path = OUT / f"twitter_test_part{shard}.jsonl"
    # Columns already computed by the first test run (same outbreaks, same observers) are reused.
    earlier = {}
    for f in OUT.glob("twitter_test_v1_part*.jsonl"):
        for r in map(json.loads, f.open()):
            r["Mean-field, one source rate"] = r.pop("Mean-field, fitted")
            r["Mean-field, one source rate seconds"] = r.pop("Mean-field, fitted seconds")
            earlier[r["source"]] = r
    done = sum(1 for _ in path.open()) if path.exists() else 0
    kept = 0
    for source in sources:
        if kept >= count:
            break
        built = outbreak(follows, either, source, cascades[source])
        if built is None:
            continue
        kept += 1
        if kept % shards != shard:
            continue
        done -= 1
        if done >= 0:
            continue
        ctx, src = built
        row = {"source": int(source), "people": ctx.graph.n_nodes, "infected observers": int(len(ctx.infected))}
        for label, fn in methods.items():
            if label in earlier.get(int(source), {}):
                row[label], row[label + " seconds"] = earlier[int(source)][label], earlier[int(source)][label + " seconds"]
                continue
            start = time.time()
            row[label] = rank_of_source(fn(ctx), src)
            row[label + " seconds"] = time.time() - start
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(int(sys.argv[2]))
    else:
        test(int(sys.argv[2]), *(int(a) for a in sys.argv[3:5]))

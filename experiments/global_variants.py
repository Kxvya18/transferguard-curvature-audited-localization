"""Real epidemics: SARS-CoV-2 variants spreading over the world air-travel network.

Public data, in data/raw/global/:
    airports.dat, routes.dat            OpenFlights (github.com/jpatokal/openflights)
    covariants/<variant>.json           CoVariants (github.com/hodcroftlab/covariants,
                                        cluster_tables/<variant>_data.json): per country and
                                        two-week period, sequences of the variant and all sequences

Outbreaks are the variants whose earliest documented samples WHO attributes to a single
country, plus 20A.EU1 (Hodcroft et al., Nature 2021). Eta and Omicron (WHO: "multiple
countries") are left out. These rules were fixed before any method was run.

Network: countries, linked by the number of airline routes between them (each route is
one contact per two-week window). A country's report is the first period in which it
sequenced the variant; countries that were sequencing but never found it by the
observation time report "not detected". Observation is 6 periods after the first
detection anywhere; the start may be up to 10 periods before it.

Mean field runs in days, while reports come per two-week period ("first sequenced
some time in that period"). It adds an observation model: once infected, a country
detects the variant on a day with probability 1 - exp(-kappa * sequences run that day),
and, infected or not, wrongly reports it with probability 1 - exp(-phi * sequences run)
(misassigned or misdated sequences). Its parameters (rate per route per day, outside
introductions per day, kappa, phi) are fitted by maximum likelihood at the true origins
of the other outbreaks only (leave one outbreak out). "Mean field,
untrained" averages over the same grid with no fitting at all. Every baseline gets the
same reports; LPSI's alpha is tuned leave one out too.

Run:  python -m experiments.global_variants
"""

from __future__ import annotations

import csv
import itertools
import json
import os
from multiprocessing import get_context
from datetime import date

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.special import logsumexp

from experiments.benchmark_localization import PROJECT_ROOT, rank_of_source
from localization import estimators as est
from localization.mean_field import mean_field_loglik
from localization.spread import Outbreak
from localization.temporal_graph import TemporalGraph

RAW = PROJECT_ROOT / "data" / "raw" / "global"
ORIGINS = {"20A.EU1": "Spain", "20H.Beta.V2": "South Africa", "20I.Alpha.V1": "United Kingdom",
           "20J.Gamma.V3": "Brazil", "21A.Delta": "India", "21B.Kappa": "India", "21C.Epsilon": "United States",
           "21F.Iota": "United States", "21G.Lambda": "Peru", "21H.Mu": "Colombia"}
RENAME = {"USA": "United States", "Sint Maarten": "Netherlands Antilles"}
OBSERVE, LOOKBACK, DAYS = 6, 10, 14  # periods after the first detection; periods before it the start may be; days per period
GRID = list(itertools.product((1e-5, 3e-5, 1e-4, 3e-4, 1e-3), (0.0, 1e-4, 1e-3), (1e-3, 1e-2, 1e-1), (0.0, 1e-5, 1e-4, 1e-3)))
LPSI_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)


def network():
    """Country names and route pairs (one row per route between two different countries)."""
    country = {row[0]: row[3] for row in csv.reader(open(RAW / "airports.dat", encoding="utf-8"))}
    names = sorted(set(country.values()))
    index = {c: i for i, c in enumerate(names)}
    pairs = []
    for row in csv.reader(open(RAW / "routes.dat", encoding="utf-8")):
        a, b = country.get(row[3]), country.get(row[5])
        if a and b and a != b:
            pairs.append(sorted((index[a], index[b])))
    return names, np.array(pairs, dtype=np.int64)


def sequencing():
    """Per variant {country: {period: variant sequences}}, and all sequences run {country: {period: n}}."""
    found, effort = {}, {}
    for variant in ORIGINS:
        data = json.load(open(RAW / "covariants" / f"{variant}.json"))
        found[variant] = {}
        for name, v in data.items():
            name = RENAME.get(name, name)
            found[variant][name] = dict(zip(v["week"], v["cluster_sequences"]))
            for week, total in zip(v["week"], v["total_sequences"]):
                effort.setdefault(name, {})[week] = max(effort.get(name, {}).get(week, 0), total)
    return found, effort


def outbreak(variant, names, pairs, found, effort):
    """Context, true origin index, and detection chance per observer per period (as a function of kappa)."""
    weeks = sorted({w for c in effort.values() for w in c})
    assert all((date.fromisoformat(b) - date.fromisoformat(a)).days == 14 for a, b in zip(weeks, weeks[1:]))
    period = {w: i for i, w in enumerate(weeks)}
    index = {c: i for i, c in enumerate(names)}
    first = {c: min(period[w] for w, k in seen.items() if k > 0) for c, seen in found[variant].items() if any(seen.values())}
    t_first = min(first.values())
    t_lo, t_obs = max(t_first - LOOKBACK, 0), t_first + OBSERVE
    runs = np.zeros((len(names), len(weeks)))
    for c, per in effort.items():
        if c in index:
            for w, n in per.items():
                runs[index[c], period[w]] = n
    observers = np.flatnonzero(runs[:, t_lo:t_obs].sum(axis=1) > 0)
    reported = np.array([min(first.get(names[i], t_obs), t_obs) for i in observers], dtype=float) * DAYS
    graph = TemporalGraph(variant, len(names), [pairs] * (t_obs * DAYS + 1), np.zeros(len(names), dtype=int))
    ob = Outbreak(index[ORIGINS[variant]], t_lo * DAYS, t_obs * DAYS, observers, reported)
    ctx = est.build_context(graph, ob, (t_obs - t_lo) * DAYS)
    daily = np.repeat(runs[observers], DAYS, axis=1) / DAYS  # sequences per day
    return ctx, index[ORIGINS[variant]], daily


def model(ctx, runs, rate, leak, kappa, phi, candidates=None):
    detection = (1.0 - np.exp(-kappa * runs)).astype(np.float32)
    false_report = (1.0 - np.exp(-phi * runs)).astype(np.float32)
    return mean_field_loglik(ctx, prob_grid=(rate,), leak=leak, floor=1e-12, weighted=True, prune=False,
                             n_start_times=LOOKBACK + 1, detection=detection, false_report=false_report,
                             report_width=DAYS, candidates=candidates)


def effective_distance(ctx, names, pairs):
    """Brockmann & Helbing (Science 2013): the source is the node from which effective
    distance D = 1 - log(share of departing traffic) best orders the arrival times."""
    n = len(names)
    w = sp.coo_matrix((np.ones(2 * len(pairs)), (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                      shape=(n, n)).tocsr()
    share = sp.diags(1.0 / np.maximum(np.asarray(w.sum(axis=1)).ravel(), 1)) @ w
    share.data = 1.0 - np.log(share.data)
    distance = dijkstra(share, directed=True)
    detected = ctx.infected
    times = ctx.infected_times
    score = np.full(n, est.INFEASIBLE)
    for s in range(n):
        d = distance[s, detected]
        ok = np.isfinite(d)
        if ok.sum() >= 3 and np.std(d[ok]) > 0 and np.std(times[ok]) > 0:
            score[s] = np.corrcoef(d[ok], times[ok])[0, 1]
    return score


CASES = {}


def truth_loglik(v):
    """Log-likelihood of outbreak v at its true origin, start summed out, for every model."""
    ctx, src, runs = CASES[v]
    return np.array([logsumexp(model(ctx, runs, *g, candidates=np.array([src]))[:, src]) for g in GRID])


def untrained(v):
    """Scores averaged over every model in the grid, with no fitting."""
    ctx, _, runs = CASES[v]
    return logsumexp(np.array([logsumexp(model(ctx, runs, *g), axis=0) for g in GRID]), axis=0)


def main() -> None:
    names, pairs = network()
    found, effort = sequencing()
    CASES.update({v: outbreak(v, names, pairs, found, effort) for v in ORIGINS})
    cases = CASES
    with get_context("fork").Pool(int(os.environ.get("WORKERS", 6))) as pool:  # workers share CASES
        at_truth = dict(zip(ORIGINS, pool.map(truth_loglik, list(ORIGINS))))
        averaged = dict(zip(ORIGINS, pool.map(untrained, list(ORIGINS))))
    lpsi_mrr = {v: {a: 1 / rank_of_source(est.lpsi(ctx, alpha=a), src) for a in LPSI_ALPHAS} for v, (ctx, src, _) in cases.items()}
    out = PROJECT_ROOT / "outputs" / "real_epidemics"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for v, (ctx, src, runs) in cases.items():
        others = [u for u in ORIGINS if u != v]
        best = GRID[int(np.argmax(sum(at_truth[u] for u in others)))]
        alpha = max(LPSI_ALPHAS, key=lambda a: np.mean([lpsi_mrr[u][a] for u in others]))
        fitted = logsumexp(model(ctx, runs, *best), axis=0)
        earliest = np.full(len(names), est.INFEASIBLE)
        earliest[ctx.infected] = -ctx.infected_times
        finish = lambda s: np.where(np.isfinite(s), s, est.INFEASIBLE)
        row = {"variant": v, "origin": names[src], "countries reporting": int(len(ctx.outbreak.observers)),
               "countries detected": int(len(ctx.infected)), "fitted model": dict(zip(("rate", "leak", "kappa", "phi"), best)),
               "Mean field, fitted": rank_of_source(finish(fitted), src),
               "Mean field, untrained": rank_of_source(finish(averaged[v]), src),
               "Earliest detection": rank_of_source(earliest, src),
               "Effective distance (Brockmann-Helbing)": rank_of_source(effective_distance(ctx, names, pairs), src),
               "LPSI, tuned": rank_of_source(est.lpsi(ctx, alpha=alpha), src),
               "Pinto-style": rank_of_source(est.pinto_style(ctx), src),
               "Jordan center": rank_of_source(est.jordan_center(ctx), src),
               "Distance center": rank_of_source(est.distance_center(ctx), src)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "variants.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()

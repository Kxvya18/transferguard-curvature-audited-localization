"""Real epidemics whose origin is known independently of who reported first: H1N1 2009
(Mexico) and SARS 2003 (China). Protocol fixed in docs/preregistration_real_outbreaks.md
(addendum of 2026-11-22) before any method was run.

Data, in data/raw/heldout/ (open, from WHO reports via Kaggle):
    h1n1/data.csv                                                        H1N1, 24 Apr - 6 Jul 2009
    sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv   SARS, 2003
Network: OpenFlights routes between countries (as in experiments/global_variants.py).

Run:  python -m experiments.independent_origins
"""

from __future__ import annotations

import itertools
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from experiments.benchmark_localization import PROJECT_ROOT, rank_of_source
from experiments.global_variants import effective_distance, network
from localization import estimators as est
from localization.mean_field import mean_field_loglik
from localization.spread import Outbreak
from localization.temporal_graph import TemporalGraph

RAW = PROJECT_ROOT / "data" / "raw" / "heldout"
OUTBREAKS = {  # name: (file, cases column, origin)
    "H1N1 2009": ("h1n1/data.csv", "Cumulative no. of cases", "Mexico"),
    "SARS 2003": ("sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv", "Cumulative number of case(s)", "China"),
}
# WHO names -> OpenFlights names (anything not listed and not in OpenFlights is dropped)
NAMES = {
    "United States of America": "United States", "UK": "United Kingdom", "Viet Nam": "Vietnam",
    "Hong Kong Special Administrative Region": "Hong Kong", "Hong Kong SAR, China": "Hong Kong",
    "Macao SAR, China": "Macau", "Taiwan, China": "Taiwan", "Republic of Korea": "South Korea",
    "Republic of Ireland": "Ireland", "Russian Federation": "Russia", "Brunei Darussalam": "Brunei",
    "Morocco ": "Morocco", "Cap Verde": "Cape Verde", "Bosnia and Hezegovina": "Bosnia and Herzegovina",
    "The former Yugoslav Republic of Macedonia": "Macedonia", "Iran, Islamic Republic": "Iran",
    "Cayman Islands, UKOT": "Cayman Islands", "Bermuda, UKOT": "Bermuda",
    "British Virgin Islands, UKOT": "British Virgin Islands", "British Virgin Islands,  UKOT": "British Virgin Islands",
    "United Kingdom, Jersey, Crown Dependency": "Jersey", "Jersey, Crown Dependency": "Jersey",
    "United Kingdom, Isle of Man, Crown Dependency": "Isle of Man", "Isle of Man, Crown Dependency": "Isle of Man",
    "Guernsey, Crown Dependency": "Guernsey", "France, French Polynesia, FOC": "French Polynesia",
    "French Polynesia, FOC": "French Polynesia", "France, Martinique, FOC": "Martinique", "Martinique, FOC": "Martinique",
    "France, New Caledonia, FOC": "New Caledonia", "New Caledonia, FOC": "New Caledonia", "Guadaloupe, FOC": "Guadeloupe",
    "Netherlands, Curacao, OT": "Netherlands Antilles", "Netherlands Antilles, CuraÃ§ao ": "Netherlands Antilles",
    "Netherlands Antilles, CuraÃ§ao": "Netherlands Antilles", "Netherlands Antilles, Sint Maarten": "Netherlands Antilles",
    "Netherlands, Aruba": "Aruba", "Cook Island": "Cook Islands", "West Bank and Gaza Strip": "Palestine",
}
OBSERVE, LOOKBACK = 28, 150  # days after the first report; days before it the start may be
GRID = list(itertools.product((1e-5, 1e-4, 1e-3), (0.0, 1e-5, 1e-4), (0.02, 0.1, 0.5)))  # rate, leak, report chance per day


def outbreak(name, names, pairs):
    path, column, origin = OUTBREAKS[name]
    data = pd.read_csv(RAW / path, encoding="latin1")
    data["Country"] = data["Country"].map(lambda c: NAMES.get(c, c))
    first = data[data[column] > 0].groupby("Country")["Date"].min()
    first = first[first.index.isin(names)]
    day0 = date.fromisoformat(first.min())
    index = {c: i for i, c in enumerate(names)}
    t_first, t_obs = LOOKBACK, LOOKBACK + OBSERVE
    reported = np.full(len(names), float(t_obs))
    for c, d in first.items():
        reported[index[c]] = min(LOOKBACK + (date.fromisoformat(d) - day0).days, t_obs)
    observers = np.arange(len(names))
    graph = TemporalGraph(name, len(names), [pairs] * (t_obs + 1), np.zeros(len(names), dtype=int))
    ctx = est.build_context(graph, Outbreak(index[origin], 0, t_obs, observers, reported), t_obs)
    return ctx, index[origin], len(first)


def mean_field(ctx):
    """Untrained: average over the fixed grid."""
    terms = []
    for rate, leak, chance in GRID:
        detection = np.full((len(ctx.outbreak.observers), ctx.outbreak.t_observe + 1), chance, dtype=np.float32)
        loglik = mean_field_loglik(ctx, prob_grid=(rate,), leak=leak, floor=1e-12, weighted=True, prune=False,
                                   n_start_times=11, detection=detection)
        terms.append(logsumexp(loglik, axis=0))
    score = logsumexp(np.array(terms), axis=0)
    return np.where(np.isfinite(score), score, est.INFEASIBLE)


def main() -> None:
    names, pairs = network()
    rows = []
    for name in OUTBREAKS:
        ctx, src, reporting = outbreak(name, names, pairs)
        earliest = np.full(len(names), est.INFEASIBLE)
        earliest[ctx.infected] = -ctx.infected_times
        row = {"outbreak": name, "origin": names[src], "countries with cases": reporting,
               "countries reporting by observation day": int(len(ctx.infected)),
               "first reporters": [names[i] for i in ctx.infected[ctx.infected_times == ctx.infected_times.min()]],
               "Mean field (untrained)": rank_of_source(mean_field(ctx), src),
               "Earliest detection": rank_of_source(earliest, src),
               "Distance center": rank_of_source(est.distance_center(ctx), src),
               "Jordan center": rank_of_source(est.jordan_center(ctx), src),
               "LPSI": rank_of_source(est.lpsi(ctx, alpha=0.5), src),
               "Pinto-style": rank_of_source(est.pinto_style(ctx), src),
               "Effective distance (Brockmann-Helbing)": rank_of_source(effective_distance(ctx, names, pairs), src)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = PROJECT_ROOT / "outputs" / "real_epidemics" / "independent_origins.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()

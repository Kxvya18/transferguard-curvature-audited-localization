"""Single-source localization benchmark on evolving networks with sparse, timed observers.

Protocol (follows Sterchi et al., 2025, adapted to sparse observers):
- one source per outbreak, SI spreading over time windows, observed `horizon`
  windows after the start; a random fraction of nodes are observers that report
  their infection time, or that they are still healthy;
- learned models are trained on outbreaks simulated on the same network
  with the true transmission probability; the mean-field localizer and the
  simulation-based methods are not told it (they sum or sample over 0.1 to 0.9);
- test outbreaks are drawn at the true rate and, to check robustness when the
  spreading model is misspecified, at a lower (0.4x) and a higher (1.6x, capped
  at 0.9) rate; the learned models are not retrained for the shifted tests;
- metrics: top-1 and top-5 accuracy, reciprocal rank, error distance (hops from
  the top-ranked node to the true source), and 90% credible-set size and
  coverage for methods that output a distribution;
- every setting is repeated over independent seeds (new outbreaks, new
  training data, new initialisation); summaries give the mean over seeds with
  a 95% t-interval.

Run:  python -m experiments.benchmark_localization --workers 6 (full run)
      python -m experiments.benchmark_localization --smoke    (quick check)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy import stats
from scipy.special import softmax

from localization import estimators as est
from localization.learned import predict, train_model
from localization.mean_field import mean_field_scores
from localization.spread import simulate_outbreak
from localization.temporal_graph import growing_network, load_netzschleuder_contacts, load_snap_temporal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "temporal"
CONTACTS = PROJECT_ROOT / "data" / "raw" / "sociopatterns"
HOUR = 3_600
DAY = 86_400
LEARNED = ("GCNSI", "Temporal GCN", "Arrival GNN")

DATASETS = {
    # name: (loader, transmission prob, horizon, lookback) in windows
    "Growing PA": (lambda seed: growing_network(500, 60, np.random.default_rng(seed)), 0.5, 10, 20),
    "CollegeMsg": (lambda seed: load_snap_temporal(RAW / "CollegeMsg.txt.gz", "CollegeMsg", DAY), 0.5, 14, 28),
    "Email-Eu": (lambda seed: load_snap_temporal(RAW / "email-Eu-core-temporal.txt.gz", "Email-Eu", DAY), 0.3, 7, 14),
    # SocioPatterns face-to-face contacts (Netzschleuder copies); windows without contacts are dropped.
    # Rates and horizons are set so that a typical outbreak reaches roughly 30 to 50% of people.
    "Hospital": (lambda seed: load_netzschleuder_contacts(CONTACTS / "sp_hospital.csv.zip", "Hospital", HOUR // 2), 0.3, 12, 24),
    "Primary school": (lambda seed: load_netzschleuder_contacts(CONTACTS / "sp_primary_school.csv.zip", "Primary school", HOUR // 4), 0.3, 8, 16),
    "Office": (lambda seed: load_netzschleuder_contacts(CONTACTS / "sp_office.csv.zip", "Office", HOUR), 0.5, 24, 48),
    "High school": (lambda seed: load_netzschleuder_contacts(CONTACTS / "sp_high_school_2012.csv.zip", "High school", HOUR), 0.5, 16, 32),
}


def rank_of_source(score: np.ndarray, source: int) -> float:
    """1-based rank of the true source; ties share the average rank."""
    above = np.sum(score > score[source])
    tied = np.sum(score == score[source]) - 1
    return float(1 + above + 0.5 * tied)


def credible_set(probability: np.ndarray, source: int, mass: float = 0.9) -> tuple[int, bool]:
    order = np.argsort(-probability)
    cumulative = np.cumsum(probability[order])
    size = int(np.searchsorted(cumulative, mass) + 1)
    return size, bool(source in order[:size])


def evaluate(ctx: est.Context, score: np.ndarray, probability: np.ndarray | None) -> dict:
    source = ctx.outbreak.source
    rank = rank_of_source(score, source)
    top = int(np.argmax(score))
    try:
        distance = nx.shortest_path_length(ctx.static, top, source)
    except nx.NetworkXNoPath:
        distance = np.nan
    row = {"rank": rank, "top1": float(rank <= 1), "top5": float(rank <= 5), "rr": 1.0 / rank, "error_distance": float(distance)}
    if probability is not None:
        size, covered = credible_set(probability, source)
        row.update({"cred90_size": size, "cred90_covered": float(covered)})
    return row


def classical_methods(ctx: est.Context, rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    static_fit = est.pinto_style(ctx)
    mean_field = mean_field_scores(ctx)
    return {
        "Mean-field (ours)": (mean_field, softmax(mean_field)),
        "Jordan center": (est.jordan_center(ctx), None),
        "Distance center": (est.distance_center(ctx), None),
        "Pinto-style TOA": (static_fit, None),
        "Time-respecting TOA": (est.temporal_screen(ctx), None),
        "LPSI": (est.lpsi(ctx), None),
        "SME": _posterior(est.sme_posterior(ctx, rng, static_fit)),
        "Simulation posterior": _posterior(est.boltzmann_posterior(ctx, rng)),
    }


def _posterior(post: est.Posterior):
    return post.score, post.probability


def run_setting(name: str, fraction: float, seed: int, args) -> list[dict]:
    loader, p, horizon, lookback = DATASETS[name]
    graph = loader(seed)
    rng = np.random.default_rng(10_000 * seed + int(fraction * 1000))

    def contexts(count: int, rate: float) -> list[est.Context]:
        out = []
        while len(out) < count:
            ob = simulate_outbreak(graph, rng, rate, horizon, fraction)
            if ob is not None:
                out.append(est.build_context(graph, ob, lookback))
        return out

    train = contexts(args.n_train, p)
    learned = {m: train_model(m, train, seed=seed, epochs=args.epochs) for m in LEARNED}

    rows = []
    for test_rate, rate in (("true", p), ("low", 0.4 * p), ("high", min(0.9, 1.6 * p))):
        for i, ctx in enumerate(contexts(args.n_test, rate)):
            results = classical_methods(ctx, rng)
            for m, model in learned.items():
                logits = predict(m, model, ctx)
                results[m] = (logits, softmax(logits))
            for method, (score, prob) in results.items():
                rows.append({"dataset": name, "observer_fraction": fraction, "seed": seed, "test_rate": test_rate,
                             "trial": i, "method": method, "n_observers": len(ctx.outbreak.observers),
                             "n_infected_observers": len(ctx.infected), **evaluate(ctx, score, prob)})
    return rows


def _run_in_worker(name: str, fraction: float, seed: int, args) -> list[dict]:
    torch.set_num_threads(1)
    return run_setting(name, fraction, seed, args)


def summarize(rows: list[dict]) -> list[dict]:
    """Mean over seeds of each seed's average, with a 95% t-interval."""
    def key(r):
        return r["dataset"], r["observer_fraction"], r["test_rate"], r["method"]

    out = []
    for dataset, fraction, test_rate, method in sorted({key(r) for r in rows}):
        sel = [r for r in rows if key(r) == (dataset, fraction, test_rate, method)]
        seeds = sorted({r["seed"] for r in sel})
        entry = {"dataset": dataset, "observer_fraction": fraction, "test_rate": test_rate, "method": method,
                 "n_seeds": len(seeds), "n_trials": len(sel)}
        for metric in ("top1", "top5", "rr", "error_distance", "cred90_size", "cred90_covered"):
            per_seed = []
            for s in seeds:
                vals = [r[metric] for r in sel if r["seed"] == s and metric in r and not np.isnan(r[metric])]
                if vals:
                    per_seed.append(float(np.mean(vals)))
            if not per_seed:
                continue
            mean = float(np.mean(per_seed))
            half = float(stats.t.ppf(0.975, len(per_seed) - 1) * stats.sem(per_seed)) if len(per_seed) > 1 else float("nan")
            entry[metric] = mean
            entry[f"{metric}_ci95"] = [mean - half, mean + half]
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105])
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--workers", type=int, default=1, help="settings run in parallel processes")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "benchmark")
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    args = parser.parse_args()
    if args.smoke:
        args.datasets, args.fractions, args.seeds = ["Growing PA"], [0.1], [1]
        args.n_train, args.n_test, args.epochs = 20, 5, 2
        args.out = args.out.parent / "benchmark_smoke"

    args.out.mkdir(parents=True, exist_ok=True)
    trials_path = args.out / "trials.csv"
    rows: list[dict] = []
    done = set()
    if trials_path.exists():  # resume an interrupted run
        with trials_path.open() as handle:
            for r in csv.DictReader(handle):
                r = {k: (float(v) if k not in ("dataset", "method", "test_rate") and v != "" else v) for k, v in r.items()}
                r = {k: v for k, v in r.items() if v != ""}
                rows.append(r)
                done.add((r["dataset"], float(r["observer_fraction"]), int(r["seed"])))

    fields = ["dataset", "observer_fraction", "seed", "test_rate", "trial", "method", "n_observers", "n_infected_observers",
              "rank", "top1", "top5", "rr", "error_distance", "cred90_size", "cred90_covered"]
    todo = [(name, fraction, seed) for name in args.datasets for fraction in args.fractions for seed in args.seeds
            if (name, fraction, seed) not in done]
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(_run_in_worker, *setting, args): setting for setting in todo}
        for job in concurrent.futures.as_completed(jobs):
            name, fraction, seed = jobs[job]
            new = job.result()
            rows.extend(new)
            write_header = not trials_path.exists()
            with trials_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                writer.writerows(new)
            mrr = {m: np.mean([r["rr"] for r in new if r["method"] == m]) for m in ("Mean-field (ours)", "Temporal GCN")}
            print(f"{name} f={fraction} seed={seed} done at {time.time() - start:.0f}s: "
                  + ", ".join(f"{m} MRR={v:.3f}" for m, v in mrr.items()), flush=True)

    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps({"protocol": __doc__, "results": summary}, indent=2))
    print(f"wrote {trials_path} and summary.json ({len(rows)} trial rows)")


if __name__ == "__main__":
    main()

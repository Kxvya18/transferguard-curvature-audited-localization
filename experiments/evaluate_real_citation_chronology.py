"""Real timestamped citation-network validation for the reliability paper.

This experiment uses the SNAP cit-HepPh graph because each paper has a public
submission date. The task is deliberately narrower than IoT compromise-source
localization: given a final citation graph, can structural evidence recover
early/source-age papers and chronological order better than standard topology
baselines?
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import kendalltau, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from priors.boltzmann_prior import (
    beta_estimate_reliability,
    graph_concentration_features,
    mle_beta,
    origin_prior_scores,
)


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cit_hepph"
OUT_DIR = PROJECT_ROOT / "outputs" / "real_citation_chronology"
FIG_DIR = PROJECT_ROOT / "paper_assets" / "figures"


def _read_dates(path: Path) -> dict[int, date]:
    dates: dict[int, date] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            node_s, date_s = line.split()
            year, month, day = [int(part) for part in date_s.split("-")]
            dates[int(node_s)] = date(year, month, day)
    return dates


def _read_edges(path: Path) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            src_s, dst_s = line.split()
            edges.append((int(src_s), int(dst_s)))
    return edges


def _chronological_nodes(dates: dict[int, date]) -> list[int]:
    return [node for node, _ in sorted(dates.items(), key=lambda item: (item[1], item[0]))]


def _build_prefix_graph(
    nodes: list[int],
    edges: Iterable[tuple[int, int]],
    dates: dict[int, date],
    n_nodes: int,
) -> tuple[nx.Graph, nx.DiGraph, dict[int, list[int]], dict[int, int]]:
    prefix_nodes = set(nodes[:n_nodes])
    order_index = {node: idx for idx, node in enumerate(nodes[:n_nodes])}
    cited_by_new: dict[int, list[int]] = defaultdict(list)
    graph = nx.Graph()
    digraph = nx.DiGraph()
    graph.add_nodes_from(prefix_nodes)
    digraph.add_nodes_from(prefix_nodes)
    for citing, cited in edges:
        if citing not in prefix_nodes or cited not in prefix_nodes:
            continue
        if dates[cited] >= dates[citing]:
            continue
        cited_by_new[citing].append(cited)
        graph.add_edge(citing, cited)
        # Direction is preserved for citation-specific baselines: newer paper
        # points to older cited paper.
        digraph.add_edge(citing, cited)
    largest = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest).copy()
    digraph = digraph.subgraph(largest).copy()
    kept = set(graph.nodes())
    cited_by_new = {
        node: [target for target in targets if target in kept]
        for node, targets in cited_by_new.items()
        if node in kept
    }
    order_index = {node: rank for rank, node in enumerate(nodes) if node in kept}
    return graph, digraph, cited_by_new, order_index


def _history_from_real_order(
    ordered_nodes: list[int],
    cited_by_new: dict[int, list[int]],
    order_index: dict[int, int],
    min_seed_nodes: int,
    max_steps: int,
) -> list[dict]:
    active: set[int] = set()
    in_degree: dict[int, int] = {}
    history: list[dict] = []
    kept_order = [node for node in ordered_nodes if node in order_index]
    for node in kept_order[:min_seed_nodes]:
        active.add(node)
        in_degree[node] = 0
    for node in kept_order[min_seed_nodes:]:
        targets = [target for target in cited_by_new.get(node, []) if target in active]
        if targets:
            # Add-one attractiveness lets uncited but active papers remain eligible.
            deg_before = {existing: in_degree.get(existing, 0) + 1 for existing in active}
            history.append(
                {
                    "new_node": node,
                    "targets": targets,
                    "deg_before": deg_before,
                    "m": len(targets),
                    "source": "snap_cit_hepph_real_dates",
                }
            )
            for target in targets:
                in_degree[target] = in_degree.get(target, 0) + 1
        active.add(node)
        in_degree.setdefault(node, 0)
        if len(history) >= max_steps:
            break
    return history


def _rank_metrics(
    scores: dict[int, float],
    order_index: dict[int, int],
    rng: np.random.Generator,
) -> dict[str, float]:
    nodes = [node for node in scores if node in order_index]
    if len(nodes) < 5:
        return {}
    predicted = np.asarray([scores[node] for node in nodes], dtype=float)
    true_oldness = np.asarray([-order_index[node] for node in nodes], dtype=float)
    tau = kendalltau(predicted, true_oldness, nan_policy="omit").statistic
    rho = spearmanr(predicted, true_oldness, nan_policy="omit").statistic
    if math.isnan(tau):
        tau = 0.0
    if math.isnan(rho):
        rho = 0.0

    ordered_pred = sorted(nodes, key=lambda node: scores[node], reverse=True)
    ordered_true = sorted(nodes, key=lambda node: order_index[node])
    random_order = list(nodes)
    rng.shuffle(random_order)

    results = {
        "kendall_tau": float(tau),
        "spearman_rho": float(rho),
        "n_eval_nodes": float(len(nodes)),
    }
    for frac in (0.01, 0.05, 0.10):
        k_val = max(1, int(round(frac * len(nodes))))
        true_top = set(ordered_true[:k_val])
        pred_top = set(ordered_pred[:k_val])
        rand_top = set(random_order[:k_val])
        results[f"oldest_precision@{int(frac * 100)}pct"] = float(len(pred_top & true_top) / k_val)
        results[f"random_oldest_precision@{int(frac * 100)}pct"] = float(len(rand_top & true_top) / k_val)
    return results


def _robust_z(scores: dict[int, float]) -> dict[int, float]:
    nodes = list(scores)
    values = np.asarray([scores[node] for node in nodes], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= 1e-9:
        scale = float(np.std(values))
    if scale <= 1e-9:
        return {node: 0.0 for node in nodes}
    return {node: float(np.clip((scores[node] - median) / scale, -6.0, 6.0)) for node in nodes}


def _baseline_scores(
    graph: nx.Graph,
    digraph: nx.DiGraph,
    beta_hat: float,
    reliability: float,
) -> dict[str, dict[int, float]]:
    degree = {node: float(graph.degree(node)) for node in graph.nodes()}
    citation_indegree = {node: float(digraph.in_degree(node)) for node in graph.nodes()}
    citation_outdegree = {node: -float(digraph.out_degree(node)) for node in graph.nodes()}
    citation_boltzmann = {
        node: float(beta_hat * np.log1p(citation_indegree[node]))
        for node in graph.nodes()
    }
    pagerank = nx.pagerank(graph, alpha=0.85, max_iter=200, tol=1e-8)
    core = nx.core_number(graph)
    clustering_inverse = {node: 1.0 - float(nx.clustering(graph, node)) for node in graph.nodes()}
    boltzmann = origin_prior_scores(graph, beta=beta_hat, m=2)
    rel_boltzmann = {
        node: float(reliability * boltzmann[node] + (1.0 - reliability) * degree[node])
        for node in graph.nodes()
    }
    composite_terms = [
        _robust_z(citation_boltzmann),
        _robust_z(pagerank),
        _robust_z(clustering_inverse),
    ]
    structural_ensemble = {
        node: float(np.mean([term[node] for term in composite_terms]))
        for node in graph.nodes()
    }
    return {
        "CitationBoltzmann": citation_boltzmann,
        "CitationInDegree": citation_indegree,
        "CitationOutDegree": citation_outdegree,
        "StructuralEnsemble": structural_ensemble,
        "Boltzmann": boltzmann,
        "ReliabilityBoltzmann": rel_boltzmann,
        "Degree": degree,
        "PageRank": {node: float(value) for node, value in pagerank.items()},
        "Core": {node: float(value) for node, value in core.items()},
        "LowClustering": clustering_inverse,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(prefix_rows: list[dict], metric_rows: list[dict], out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    prefix_n = np.asarray([row["prefix_nodes"] for row in prefix_rows], dtype=float)
    beta = np.asarray([row["beta_hat"] for row in prefix_rows], dtype=float)
    fisher = np.asarray([row["fisher_information"] for row in prefix_rows], dtype=float)
    reliability = np.asarray([row["reliability"] for row in prefix_rows], dtype=float)
    edges = np.asarray([row["edges"] for row in prefix_rows], dtype=float)

    axes[0, 0].plot(prefix_n, edges, marker="o", color="#334155")
    axes[0, 0].set_title("(a) Real citation graph growth")
    axes[0, 0].set_xlabel("Chronological prefix nodes")
    axes[0, 0].set_ylabel("Edges in largest WCC")
    axes[0, 0].grid(alpha=0.25)

    ax_beta = axes[0, 1]
    ax_fisher = ax_beta.twinx()
    ax_beta.plot(prefix_n, beta, marker="o", color="#c2410c", label=r"$\hat{\beta}$")
    ax_fisher.plot(prefix_n, fisher, marker="s", color="#166534", label="Fisher")
    ax_beta.set_title(r"(b) Fitted $\beta$ and identifiability")
    ax_beta.set_xlabel("Chronological prefix nodes")
    ax_beta.set_ylabel(r"$\hat{\beta}$", color="#c2410c")
    ax_fisher.set_ylabel("Fisher information", color="#166534")
    ax_beta.grid(alpha=0.25)

    axes[0, 2].plot(prefix_n, reliability, marker="o", color="#0f766e")
    axes[0, 2].set_ylim(0.0, 1.02)
    axes[0, 2].set_title("(c) Structural reliability weight")
    axes[0, 2].set_xlabel("Chronological prefix nodes")
    axes[0, 2].set_ylabel(r"$r_F$")
    axes[0, 2].grid(alpha=0.25)

    methods = sorted({row["method"] for row in metric_rows})
    colors = {
        "CitationBoltzmann": "#ea580c",
        "CitationInDegree": "#2563eb",
        "CitationOutDegree": "#94a3b8",
        "StructuralEnsemble": "#047857",
        "Boltzmann": "#c2410c",
        "ReliabilityBoltzmann": "#0f766e",
        "Degree": "#1d4ed8",
        "PageRank": "#7c3aed",
        "Core": "#475569",
        "LowClustering": "#db2777",
    }
    for method in methods:
        rows = [row for row in metric_rows if row["method"] == method]
        x_val = [row["prefix_nodes"] for row in rows]
        y_val = [row["kendall_tau"] for row in rows]
        axes[1, 0].plot(x_val, y_val, marker="o", label=method, color=colors.get(method))
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    axes[1, 0].set_title("(d) Chronological order recovery")
    axes[1, 0].set_xlabel("Chronological prefix nodes")
    axes[1, 0].set_ylabel(r"Kendall $\tau$ vs true paper age")
    axes[1, 0].grid(alpha=0.25)

    for method in methods:
        rows = [row for row in metric_rows if row["method"] == method]
        x_val = [row["prefix_nodes"] for row in rows]
        y_val = [row["oldest_precision@5pct"] for row in rows]
        axes[1, 1].plot(x_val, y_val, marker="o", label=method, color=colors.get(method))
    axes[1, 1].set_title("(e) Oldest-paper retrieval")
    axes[1, 1].set_xlabel("Chronological prefix nodes")
    axes[1, 1].set_ylabel("Precision@oldest 5%")
    axes[1, 1].grid(alpha=0.25)

    final_rows = [row for row in metric_rows if row["prefix_nodes"] == max(prefix_n)]
    final_rows = sorted(final_rows, key=lambda row: row["kendall_tau"], reverse=True)
    y_pos = np.arange(len(final_rows))
    axes[1, 2].barh(
        y_pos,
        [row["kendall_tau"] for row in final_rows],
        color=[colors.get(row["method"], "#64748b") for row in final_rows],
    )
    axes[1, 2].set_yticks(y_pos, [row["method"] for row in final_rows])
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_title("(f) Final-prefix baseline comparison")
    axes[1, 2].set_xlabel(r"Kendall $\tau$")
    axes[1, 2].grid(axis="x", alpha=0.25)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "Real Timestamped Validation on SNAP cit-HepPh: Structural Recovery of Citation Chronology",
        y=1.04,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    raw_edges = RAW_DIR / "cit-HepPh.txt.gz"
    raw_dates = RAW_DIR / "cit-HepPh-dates.txt.gz"
    if not raw_edges.exists() or not raw_dates.exists():
        raise FileNotFoundError(
            "Missing SNAP cit-HepPh files. Download cit-HepPh.txt.gz and "
            "cit-HepPh-dates.txt.gz into data/raw/cit_hepph first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dates = _read_dates(raw_dates)
    edges = _read_edges(raw_edges)
    nodes = _chronological_nodes(dates)
    rng = np.random.default_rng(args.seed)

    prefix_rows: list[dict] = []
    metric_rows: list[dict] = []
    for prefix_size in args.prefix_sizes:
        graph, digraph, cited_by_new, order_index = _build_prefix_graph(nodes, edges, dates, prefix_size)
        ordered_kept = [node for node in nodes if node in graph]
        history = _history_from_real_order(
            ordered_kept,
            cited_by_new,
            order_index,
            min_seed_nodes=args.min_seed_nodes,
            max_steps=args.max_mle_steps,
        )
        estimate = mle_beta(
            history,
            beta_min=0.0,
            beta_max=args.beta_max,
            coarse_points=args.coarse_points,
            refine_rounds=args.refine_rounds,
        )
        reliability = beta_estimate_reliability(estimate, information_scale=args.information_scale)
        concentration = graph_concentration_features(graph, estimate.beta_hat, reliability=reliability)
        prefix_row = {
            "dataset": "SNAP cit-HepPh",
            "prefix_nodes": int(prefix_size),
            "largest_wcc_nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "history_steps": int(len(history)),
            "beta_hat": estimate.beta_hat,
            "fisher_information": estimate.fisher_information,
            "fisher_se": estimate.se,
            "reliability": reliability,
            "ci_lower": estimate.ci_lower,
            "ci_upper": estimate.ci_upper,
            "curvature_relative_error": estimate.curvature_relative_error,
            "degree_concentration": concentration["degree_concentration"],
            "prior_ambiguity": concentration["prior_ambiguity"],
            "prior_top1": concentration["prior_top1"],
        }
        prefix_rows.append(prefix_row)

        for method, scores in _baseline_scores(graph, digraph, estimate.beta_hat, reliability).items():
            row = {
                "dataset": "SNAP cit-HepPh",
                "prefix_nodes": int(prefix_size),
                "method": method,
                **_rank_metrics(scores, order_index, rng),
            }
            metric_rows.append(row)

    _write_csv(OUT_DIR / "prefix_summary.csv", prefix_rows)
    _write_csv(OUT_DIR / "chronology_metrics.csv", metric_rows)
    figure_path = FIG_DIR / "fig_real_citation_chronology.png"
    _plot_results(prefix_rows, metric_rows, figure_path)

    final_prefix = max(args.prefix_sizes)
    final_rows = [row for row in metric_rows if row["prefix_nodes"] == final_prefix]
    best = max(final_rows, key=lambda row: row["kendall_tau"])
    summary = {
        "dataset": "SNAP cit-HepPh",
        "source": "https://snap.stanford.edu/data/cit-HepPh.html",
        "raw_files": {
            "edges": str(raw_edges),
            "dates": str(raw_dates),
        },
        "protocol": {
            "task": "timestamp-supervised citation chronology reconstruction from final graph structure",
            "prefix_sizes": list(args.prefix_sizes),
            "min_seed_nodes": args.min_seed_nodes,
            "max_mle_steps": args.max_mle_steps,
            "evaluation": "Kendall tau/Spearman rho against true paper age; top-oldest retrieval precision",
        },
        "final_prefix": final_prefix,
        "best_final_method": best,
        "beta_estimates": prefix_rows,
        "figure": str(figure_path),
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-sizes", nargs="+", type=int, default=[1000, 2000, 4000, 8000])
    parser.add_argument("--min-seed-nodes", type=int, default=20)
    parser.add_argument("--max-mle-steps", type=int, default=1800)
    parser.add_argument("--beta-max", type=float, default=4.0)
    parser.add_argument("--coarse-points", type=int, default=25)
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument("--information-scale", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=1907)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))

"""Generate statistical-physics-style submission figures from persisted results."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper_assets" / "figures"


def _read(path):
    with Path(path).open() as handle: return json.load(handle)


def main() -> None:
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .22})
    fixed = _read(ROOT / "outputs/intervention_utility_v4/summary.json")
    adaptive = _read(ROOT / "outputs/adaptive_intervention_v4/summary.json")
    profile = _read(ROOT / "outputs/profiling_v4/summary.json")
    cross = {name: _read(ROOT / f"outputs/cross_dataset_v4/{name}.json") for name in ("iot23_hybrid", "iot23_temporal", "ton_hybrid", "ton_temporal")}
    fixed_map = {(x["method"], x["k"]): x["mean_prevented_fraction"] for x in fixed["intervention_aggregate"]}
    adaptive_map = {(x["method"], x["k"]): x["mean_net_prevented"] for x in adaptive["aggregate"]}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), constrained_layout=True)
    fig.suptitle("TRANSFERGUARD: Regime Dependence, Transfer, and Computational Scaling", fontsize=17)
    ax = axes[0, 0]
    methods = ["Source-only", "Decision-aware", "Degree", "Outbound"]
    fixed_values = [fixed_map[("temporal_only",3)], fixed_map[("intervention_risk_temporal",3)], fixed_map[("degree",3)], fixed_map[("outbound_activity",3)]]
    adaptive_values = [adaptive_map[("temporal_source",3)], adaptive_map[("decision_aware",3)], adaptive_map[("degree",3)], adaptive_map[("outbound_activity",3)]]
    x=np.arange(4); width=.36
    ax.bar(x-width/2, fixed_values, width, color="#2A9D8F", label="Fixed cascade")
    ax.bar(x+width/2, adaptive_values, width, color="#E76F51", label="Adaptive rerouting")
    ax.set_xticks(x, methods); ax.set_ylabel(r"containment utility $U_3$"); ax.set_title(r"(a) Intervention objective changes with attacker dynamics"); ax.legend(frameon=False)
    ax = axes[0,1]
    for method,label,color in (("temporal_source","Source-only","#264653"),("decision_aware","Decision-aware","#2A9D8F"),("degree","Degree","#E9C46A"),("outbound_activity","Outbound","#E76F51")):
        ax.plot((1,3,5),[adaptive_map[(method,k)] for k in (1,3,5)],marker="o",lw=2.2,label=label,color=color)
    ax.set_xlabel(r"quarantine budget $k$"); ax.set_ylabel(r"net prevented fraction $U_k^{\rm adapt}$"); ax.set_title("(b) Adaptive containment response"); ax.legend(frameon=False)
    ax=axes[1,0]
    labels=["IoT-23\nhybrid","IoT-23\ntemporal","TON\nhybrid","TON\ntemporal"]
    values=[cross[x]["mean_mrr"] for x in ("iot23_hybrid","iot23_temporal","ton_hybrid","ton_temporal")]
    probabilities=[cross[x]["mean_origin_probability"] for x in ("iot23_hybrid","iot23_temporal","ton_hybrid","ton_temporal")]
    ax.bar(np.arange(4),values,color=["#2A9D8F","#264653","#2A9D8F","#264653"]); ax.set_xticks(np.arange(4),labels); ax.set_ylim(0,1.08); ax.set_ylabel("zero-shot MRR"); ax.set_title("(c) Frozen synthetic-to-real transfer")
    for i,(v,p) in enumerate(zip(values,probabilities)): ax.text(i,v+.025,rf"$p_o={p:.2f}$",ha="center",fontsize=9)
    ax=axes[1,1]
    rows=profile["rows"]
    for method,label,color in (("safe_hybrid","Hybrid","#2A9D8F"),("temporal_only","Temporal","#264653")):
        subset=[x for x in rows if x["model"]==method]; ax.plot([x["n_nodes"] for x in subset],[x["latency_median_ms"] for x in subset],marker="s",lw=2,label=label,color=color)
    ax.set_xlabel(r"graph order $n$"); ax.set_ylabel("median CPU latency (ms)"); ax.set_title("(d) Inference scaling"); ax.legend(frameon=False)
    FIGURES.mkdir(parents=True,exist_ok=True); path=FIGURES/"fig_submission_physics_suite_v5.png"; fig.savefig(path,dpi=300,bbox_inches="tight"); print(path)


if __name__ == "__main__": main()

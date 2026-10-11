"""Assemble claim-safe submission tables from completed experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with Path(path).open() as handle: return json.load(handle)


def main() -> None:
    multiseed = read(ROOT / "outputs/multiseed_v4/evaluation_summary.json")
    hybrid = next(x for x in multiseed["summary"] if x["method"] == "hybrid")
    temporal = next(x for x in multiseed["summary"] if x["method"] == "temporal_only")
    reference_test = read(ROOT / "outputs/reference_baselines/fixed_test_summary.json")
    static = next(x for x in reference_test["rows"] if x["method"] == "static_gcn")
    tgat = next(x for x in reference_test["rows"] if x["method"] == "windowed_tgat")
    temporal_seed = next(
        x for x in multiseed["seed_rows"]
        if x["method"] == "temporal_only" and x["seed"] == 20260810
    )
    full = read(ROOT / "outputs/train_transferguard_mixed_no_prior_v4/training_summary.json")
    no_ssl = read(ROOT / "outputs/ablations_v4/no_ssl/training_summary.json")
    uncertainty = read(ROOT / "outputs/calibrated_selective_v4/summary.json")
    shift = read(ROOT / "outputs/cross_dataset_v4/shift_detection_summary.json")
    fixed_intervention = read(ROOT / "outputs/intervention_utility_v4/summary.json")
    adaptive_intervention = read(ROOT / "outputs/adaptive_intervention_v4/summary.json")
    cold_start = read(ROOT / "outputs/cold_start_v1/confirmatory_evaluation/summary.json")

    def intervention_value(payload, method, k, field):
        rows = payload.get("intervention_aggregate", payload.get("aggregate", []))
        return next(float(row[field]) for row in rows if row["method"] == method and row["k"] == k)

    fixed_decision = intervention_value(
        fixed_intervention, "intervention_risk_temporal", 3, "mean_prevented_fraction"
    )
    fixed_source = intervention_value(
        fixed_intervention, "temporal_only", 3, "mean_prevented_fraction"
    )
    adaptive_decision = intervention_value(
        adaptive_intervention, "decision_aware", 3, "mean_net_prevented"
    )
    adaptive_source = intervention_value(
        adaptive_intervention, "temporal_source", 3, "mean_net_prevented"
    )
    payload = {
        "multiseed": multiseed,
        "ablation_table": [
            {"component": "Boltzmann residual prior", "with_mrr": hybrid["mean_mrr"], "without_mrr": temporal["mean_mrr"], "delta": hybrid["mean_mrr"] - temporal["mean_mrr"], "basis": "3 training seeds, fixed 120-graph test"},
            {"component": "Temporal encoder vs static GCN", "with_mrr": temporal_seed["mrr"], "without_mrr": static["mrr"], "delta": temporal_seed["mrr"] - static["mrr"], "basis": "same optimizer seed and fixed 120-graph test"},
            {"component": "TRANSFERGUARD vs windowed TGAT", "with_mrr": temporal_seed["mrr"], "without_mrr": tgat["mrr"], "delta": temporal_seed["mrr"] - tgat["mrr"], "basis": "same optimizer seed and fixed 120-graph test"},
            {"component": "Self-supervised auxiliary loss", "with_mrr": full["best_val_mrr"], "without_mrr": no_ssl["best_val_mrr"], "delta": full["best_val_mrr"] - no_ssl["best_val_mrr"], "basis": "shared 144-graph validation split, one training seed"},
        ],
        "uncertainty_ablation": {item["model"]: item["ood_detection_auroc"] for item in uncertainty["models"]},
        "uncertainty_detector_control": {
            "random_detector_auroc": 0.5,
            "best_hybrid_entropy_auroc": next(item for item in uncertainty["models"] if item["model"] == "safe_hybrid")["ood_detection_auroc"]["normalized_predictive_entropy"],
            "best_temporal_mc_auroc": next(item for item in uncertainty["models"] if item["model"] == "temporal_only")["ood_detection_auroc"]["mc_mutual_information"],
            "cross_dataset_shift": shift,
        },
        "intervention_reranker_ablation_k3": {
            "fixed_cascade": {"decision_aware": fixed_decision, "source_only": fixed_source, "delta": fixed_decision - fixed_source},
            "adaptive_rerouting": {"decision_aware": adaptive_decision, "source_only": adaptive_source, "delta": adaptive_decision - adaptive_source},
            "interpretation": "The reranker helps retrospectively but reverses under adaptive rerouting.",
        },
        "confirmatory_cold_start": {
            "test_seed_base": cold_start["test_seed_base"],
            "validation_fixed_selective_policy": cold_start["validation_fixed_selective_policy"],
            "reliability_interaction": cold_start["reliability_interaction"],
            "broad_hypothesis": cold_start["pre_specified_hypothesis_test"],
            "claim": "Fisher reliability selects a minority of graphs where the inherited Boltzmann prior yields a small independently confirmed ranking gain; the universal cold-start hypothesis remains false.",
        },
        "claim_boundary": "Temporal modeling is the robust global contribution. Boltzmann fusion is useful only under validation-defined high Fisher reliability, while the current SSL objective remains unsupported.",
    }
    output = ROOT / "outputs/submission_evidence_v5"; output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w") as handle: json.dump(payload, handle, indent=2)
    print(output / "summary.json")


if __name__ == "__main__": main()

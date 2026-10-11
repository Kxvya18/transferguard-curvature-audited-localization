"""Focused regression tests for safety-critical evaluation utilities."""

from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from interventions.containment import time_respecting_quarantine
from uncertainty.calibration import (
    binary_auroc,
    classification_calibration_metrics,
    conformal_quantile,
    evaluate_prediction_sets,
)


class InterventionTests(unittest.TestCase):
    def test_blocking_frontier_prevents_chain_descendants(self) -> None:
        graph = nx.path_graph(4)
        result = time_respecting_quarantine(
            graph,
            {0: 0, 1: 1, 2: 2, 3: 3},
            quarantined=[1],
            decision_time=1,
        )
        self.assertEqual(result["baseline_future"], 2)
        self.assertEqual(result["prevented_future"], 2)
        self.assertEqual(result["prevented_fraction"], 1.0)

    def test_alternate_live_parent_preserves_future_node(self) -> None:
        graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
        result = time_respecting_quarantine(
            graph,
            {0: 0, 1: 1, 2: 1, 3: 2},
            quarantined=[1],
            decision_time=1,
        )
        self.assertEqual(result["surviving_future"], 1)
        self.assertEqual(result["prevented_fraction"], 0.0)


class CalibrationTests(unittest.TestCase):
    def test_perfect_predictions_have_zero_error(self) -> None:
        probabilities = [np.asarray([0.0, 1.0]), np.asarray([1.0, 0.0])]
        metrics = classification_calibration_metrics(probabilities, [1, 0])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertAlmostEqual(metrics["nll"], 0.0)
        self.assertAlmostEqual(metrics["brier"], 0.0)

    def test_conformal_sets_cover_calibration_labels(self) -> None:
        probabilities = [np.asarray([0.8, 0.2]), np.asarray([0.1, 0.9])]
        labels = [0, 1]
        quantile = conformal_quantile(probabilities, labels, alpha=0.1)
        result = evaluate_prediction_sets(probabilities, labels, quantile)
        self.assertEqual(result["empirical_coverage"], 1.0)

    def test_binary_auroc_handles_separated_scores(self) -> None:
        self.assertEqual(binary_auroc(np.asarray([0.1, 0.2]), np.asarray([0.8, 0.9])), 1.0)


if __name__ == "__main__":
    unittest.main()

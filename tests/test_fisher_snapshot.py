import unittest

import networkx as nx

from priors.boltzmann_prior import (
    fisher_information_approx,
    fisher_information_exact,
    history_loglikelihood_seq,
    snapshot_beta_ensemble,
    snapshot_pseudohistory,
)


class FisherCurvatureTests(unittest.TestCase):
    def test_exact_curvature_matches_finite_difference(self):
        history = [
            {
                "deg_before": {0: 8, 1: 5, 2: 3, 3: 2, 4: 1},
                "targets": [0, 2, 3],
            },
            {
                "deg_before": {0: 9, 1: 5, 2: 4, 3: 3, 4: 1, 5: 3},
                "targets": [1, 5],
            },
        ]
        beta = 1.3
        step = 1e-4
        second = (
            history_loglikelihood_seq(history, beta + step)
            - 2.0 * history_loglikelihood_seq(history, beta)
            + history_loglikelihood_seq(history, beta - step)
        ) / step**2
        self.assertAlmostEqual(fisher_information_exact(history, beta), -second, places=5)

    def test_wor_exact_differs_from_first_draw_approximation(self):
        history = [
            {
                "deg_before": {0: 20, 1: 9, 2: 4, 3: 2, 4: 1},
                "targets": [0, 1, 2, 3],
            }
        ]
        self.assertNotAlmostEqual(
            fisher_information_exact(history, 2.0),
            fisher_information_approx(history, 2.0),
            places=6,
        )


class SnapshotHistoryTests(unittest.TestCase):
    def test_snapshot_history_uses_graph_structure_only(self):
        graph = nx.barabasi_albert_graph(40, 2, seed=7)
        for node in graph:
            graph.nodes[node]["birth_time"] = 10_000 - node
        history = snapshot_pseudohistory(graph, tie_break_seed=11)
        self.assertEqual(len(history), 35)
        self.assertTrue(all(step["source"] == "snapshot_reverse_peeling" for step in history))

    def test_snapshot_ensemble_returns_finite_estimate(self):
        graph = nx.barabasi_albert_graph(50, 2, seed=13)
        estimate, diagnostics = snapshot_beta_ensemble(graph, ensemble_size=4, seed_base=17)
        self.assertGreaterEqual(estimate.beta_hat, 0.0)
        self.assertGreater(estimate.fisher_information, 0.0)
        self.assertEqual(diagnostics["ensemble_size"], 4.0)


if __name__ == "__main__":
    unittest.main()

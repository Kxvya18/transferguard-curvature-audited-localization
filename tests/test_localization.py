import unittest

import numpy as np

from localization.estimators import boltzmann_posterior, build_context, jordan_center, temporal_screen
from localization.mean_field import mean_field_marginals, mean_field_scores
from localization.spread import Outbreak, earliest_arrival, spread
from localization.temporal_graph import TemporalGraph, _join_times


def _line_graph() -> TemporalGraph:
    # 0-1 at t=0, 1-2 at t=1, 2-3 at t=2: a time-respecting path 0 -> 3.
    windows = [np.array([[0, 1]]), np.array([[1, 2]]), np.array([[2, 3]]), np.zeros((0, 2), dtype=np.int64)]
    return TemporalGraph("line", 4, windows, _join_times(4, windows))


class LocalizationTests(unittest.TestCase):
    def test_certain_spread_matches_earliest_arrival(self):
        graph = _line_graph()
        start = np.full((1, 4), np.inf)
        start[0, 0] = 0
        times = spread(graph, start, 0, 4, p=1.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(times[0], earliest_arrival(graph, 0, 4)[0])
        np.testing.assert_array_equal(times[0], [0, 1, 2, 3])

    def test_time_order_is_respected(self):
        # Node 3 can reach node 0 statically, but every contact on the way
        # happens before the infection could arrive.
        arrival = earliest_arrival(_line_graph(), 0, 4)
        self.assertTrue(np.isinf(arrival[3, 0]))

    def test_late_joiners_are_infeasible_sources(self):
        graph = _line_graph()
        outbreak = Outbreak(source=0, t_start=0, t_observe=4, observers=np.array([1, 2, 3]), observed_times=np.array([1.0, 2.0, 3.0]))
        ctx = build_context(graph, outbreak, max_lookback=4)
        score = temporal_screen(ctx)
        self.assertEqual(int(np.argmax(score)), 0)
        self.assertLess(score[3], -1e11)  # node 3 joins at t=2, after the first infection

    def test_posterior_is_normalized_and_ranks_true_source_first(self):
        # Node 0 must be observed: otherwise "node 1 starts at t=1" explains
        # the data exactly as well, and the source is not identifiable.
        graph = _line_graph()
        outbreak = Outbreak(source=0, t_start=0, t_observe=4, observers=np.array([0, 1, 2, 3]), observed_times=np.array([0.0, 1.0, 2.0, 3.0]))
        ctx = build_context(graph, outbreak, max_lookback=4)
        post = boltzmann_posterior(ctx, np.random.default_rng(0), prob_range=(0.99, 1.0), n_simulations=20)
        self.assertAlmostEqual(post.probability.sum(), 1.0, places=6)
        self.assertEqual(int(np.argmax(post.score)), 0)

    def test_mean_field_ranks_true_source_first_and_rules_out_late_joiners(self):
        graph = _line_graph()
        outbreak = Outbreak(source=0, t_start=0, t_observe=4, observers=np.array([0, 1, 2, 3]), observed_times=np.array([0.0, 1.0, 2.0, 3.0]))
        ctx = build_context(graph, outbreak, max_lookback=4)
        score = mean_field_scores(ctx)
        self.assertEqual(int(np.argmax(score)), 0)
        self.assertLess(score[3], -1e11)  # node 3 joins after the first infection

    def test_jordan_center_picks_the_middle_of_the_observers(self):
        graph = _line_graph()
        outbreak = Outbreak(source=0, t_start=0, t_observe=4, observers=np.array([0, 3]), observed_times=np.array([0.0, 3.0]))
        ctx = build_context(graph, outbreak, max_lookback=4)
        self.assertIn(int(np.argmax(jordan_center(ctx))), (1, 2))

    def test_mean_field_upper_bounds_exact_infection_probability(self):
        # Lemma 1: on a looped network with repeated contacts, mean field never falls
        # below the true SI probability (checked against 40,000 simulated outbreaks).
        rng = np.random.default_rng(3)
        n, horizon, p = 8, 12, 0.4
        edges = np.array([[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 5], [5, 3], [5, 6], [6, 7], [7, 1]])
        windows = [edges[rng.random(len(edges)) < 0.5] for _ in range(horizon)]
        graph = TemporalGraph("loops", n, windows, np.zeros(n, dtype=int))
        q = mean_field_marginals(graph, 0, 0, horizon, p)
        runs = 40_000
        start = np.full((runs, n), np.inf)
        start[:, 0] = 0
        times = spread(graph, start, 0, horizon, p, rng)
        for t in range(horizon + 1):
            exact = np.mean(times <= t, axis=0)
            margin = 4 * np.sqrt(exact * (1 - exact) / runs) + 1e-9
            self.assertTrue(np.all(q[t] >= exact - margin), (t, q[t], exact))
        self.assertGreater(np.max(q[-1] - np.mean(times <= horizon, axis=0)), 0.01)  # the bound is not tight here

    def test_pruning_never_drops_the_true_source(self):
        # Lemma 2: infection cannot beat the earliest arrival, so the true source always
        # survives the earliest-arrival screen and gets a finite mean-field score.
        from localization.spread import simulate_outbreak
        from localization.temporal_graph import growing_network
        graph = growing_network(150, 80, np.random.default_rng(5))
        rng = np.random.default_rng(11)
        checked = 0
        while checked < 25:
            ob = simulate_outbreak(graph, rng, p=0.5, horizon=15, observer_fraction=0.1)
            if ob is None:
                continue
            ctx = build_context(graph, ob, max_lookback=30)
            self.assertGreater(mean_field_scores(ctx)[ob.source], -1e11)
            checked += 1

    def test_source_rate_equal_to_rate_changes_nothing(self):
        """Giving the source its own rate, set equal to everyone else's, is the plain model."""
        from localization.mean_field import mean_field_loglik
        from localization.spread import simulate_outbreak
        from localization.temporal_graph import growing_network
        rng = np.random.default_rng(11)
        graph = growing_network(60, 40, rng)
        ob = None
        while ob is None:
            ob = simulate_outbreak(graph, rng, 0.5, 10, 0.3)
        ctx = build_context(graph, ob, 20)
        plain = mean_field_loglik(ctx, prob_grid=(0.3,))
        split = mean_field_loglik(ctx, prob_grid=(0.3,), seed_prob=0.3)
        self.assertTrue(np.allclose(np.nan_to_num(plain, neginf=-1e9), np.nan_to_num(split, neginf=-1e9), atol=1e-3))

    def test_certain_detection_is_the_plain_model(self):
        """A detection model in which every infected observer is detected at once is plain mean field."""
        from localization.mean_field import mean_field_loglik
        from localization.spread import simulate_outbreak
        from localization.temporal_graph import growing_network
        rng = np.random.default_rng(12)
        graph = growing_network(60, 40, rng)
        ob = None
        while ob is None:
            ob = simulate_outbreak(graph, rng, 0.5, 10, 0.3)
        ctx = build_context(graph, ob, 20)
        plain = mean_field_loglik(ctx, prob_grid=(0.3,))
        certain = mean_field_loglik(ctx, prob_grid=(0.3,), detection=np.ones((len(ob.observers), graph.n_windows)))
        self.assertTrue(np.allclose(np.nan_to_num(plain, neginf=-1e9), np.nan_to_num(certain, neginf=-1e9), atol=1e-3))

    def test_constant_detectability_changes_nothing(self):
        """Detectability that does not grow with time since infection is the plain detection model."""
        from localization.mean_field import mean_field_loglik
        from localization.spread import simulate_outbreak
        from localization.temporal_graph import growing_network
        rng = np.random.default_rng(13)
        graph = growing_network(60, 40, rng)
        ob = None
        while ob is None:
            ob = simulate_outbreak(graph, rng, 0.5, 10, 0.3)
        ctx = build_context(graph, ob, 20)
        detection = np.full((len(ob.observers), graph.n_windows), 0.3)
        plain = mean_field_loglik(ctx, prob_grid=(0.3,), detection=detection)
        aged = mean_field_loglik(ctx, prob_grid=(0.3,), detection=detection, detectability=np.ones(5))
        self.assertTrue(np.allclose(np.nan_to_num(plain, neginf=-1e9), np.nan_to_num(aged, neginf=-1e9), atol=1e-3))


if __name__ == "__main__":
    unittest.main()


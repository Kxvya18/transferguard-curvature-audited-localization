import unittest

import numpy as np

from localization.estimators import boltzmann_posterior, build_context, jordan_center, temporal_screen
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

    def test_jordan_center_picks_the_middle_of_the_observers(self):
        graph = _line_graph()
        outbreak = Outbreak(source=0, t_start=0, t_observe=4, observers=np.array([0, 3]), observed_times=np.array([0.0, 3.0]))
        ctx = build_context(graph, outbreak, max_lookback=4)
        self.assertIn(int(np.argmax(jordan_center(ctx))), (1, 2))


if __name__ == "__main__":
    unittest.main()

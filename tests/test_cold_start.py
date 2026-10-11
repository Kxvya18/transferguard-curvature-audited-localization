import unittest

import torch

from experiments.cold_start_common import prepare_prefix_batch
from fusion.learned_fusion import ColdStartResidualPosteriorFusion


class ColdStartProtocolTests(unittest.TestCase):
    def _batch(self):
        node_features = torch.ones((2, 6, 3, 15), dtype=torch.float32)
        structural = torch.zeros((2, 3, 7), dtype=torch.float32)
        structural[:, :, 1] = torch.tensor([0.6, 0.3, 0.1])
        structural[:, :, 2] = 0.4
        structural[:, :, 4] = 9.0
        structural[:, :, 5] = 0.25
        structural[:, :, 6] = 0.8
        mask = torch.ones((2, 3), dtype=torch.float32)
        return {
            "node_features": node_features,
            "structural_features": structural,
            "node_mask": mask,
            "ssl_mask": torch.ones((2, 6, 3), dtype=torch.bool),
        }

    def test_prefix_removes_future_aggregate_and_masks_telemetry(self):
        prepared = prepare_prefix_batch(self._batch(), 2, 0.0, 17, False)
        self.assertEqual(tuple(prepared["node_features"].shape), (2, 2, 3, 15))
        self.assertTrue(torch.all(prepared["node_features"][:, :, :, 2:8] == 0))
        self.assertTrue(torch.all(prepared["node_features"][:, :, :, :2] == 1))
        self.assertTrue(torch.all(prepared["structural_features"][:, :, 4] == 0))
        self.assertTrue(
            torch.allclose(
                prepared["observation_context"],
                torch.tensor([[2 / 6, 0.0], [2 / 6, 0.0]]),
            )
        )

    def test_prior_ablation_forces_uniform_prior_and_zero_reliability(self):
        prepared = prepare_prefix_batch(self._batch(), 1, 1.0, 23, True)
        self.assertTrue(
            torch.allclose(prepared["structural_features"][:, :, 1], torch.full((2, 3), 1 / 3))
        )
        self.assertTrue(torch.all(prepared["structural_features"][:, :, 5:7] == 0))

    def test_fisher_reliability_bounds_prior_strength(self):
        torch.manual_seed(5)
        fusion = ColdStartResidualPosteriorFusion(8, 7, dropout=0.0)
        embeddings = torch.randn((2, 3, 8))
        structural = self._batch()["structural_features"].clone()
        structural[0, :, 6] = 0.5
        structural[1, :, 6] = 0.0
        output = fusion(
            embeddings,
            structural,
            torch.ones((2, 3)),
            torch.tensor([[1 / 6, 0.25], [1.0, 1.0]]),
        )
        strength = output["prior_strength"].detach()
        self.assertGreaterEqual(float(strength[0]), 0.0)
        self.assertLessEqual(float(strength[0]), 1.0)
        self.assertEqual(float(strength[1]), 0.0)

    def test_hard_selection_disables_prior_below_threshold(self):
        torch.manual_seed(7)
        fusion = ColdStartResidualPosteriorFusion(8, 7, dropout=0.0)
        fusion.selection_threshold = 0.75
        structural = self._batch()["structural_features"].clone()
        structural[0, :, 6] = 0.74
        structural[1, :, 6] = 0.76
        output = fusion(
            torch.randn((2, 3, 8)),
            structural,
            torch.ones((2, 3)),
            torch.tensor([[1 / 6, 0.25], [1 / 6, 0.25]]),
        )
        self.assertEqual(float(output["prior_strength"][0]), 0.0)
        self.assertGreater(float(output["prior_strength"][1]), 0.0)


if __name__ == "__main__":
    unittest.main()

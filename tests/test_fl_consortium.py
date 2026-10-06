"""
Unit tests for FedSure Mutual Federated Learning Consortium Service.
"""

import os
import unittest
from src.fl_consortium import FLConsortiumService
from src.config import PROJECT_ROOT


class TestFLConsortiumService(unittest.TestCase):
    def test_consortium_overview(self):
        """Verify consortium topology, active branches, and parameters."""
        overview = FLConsortiumService.get_consortium_overview()
        self.assertIn("Pan-European", overview["consortium_name"])
        self.assertEqual(len(overview["branches"]), 4)
        self.assertEqual(overview["total_federated_nodes"], 33)
        self.assertEqual(overview["total_isolated_samples"], 51920)
        self.assertIn("active_model_checkpoint", overview)

    def test_trigger_consortium_round_simulation(self):
        """Test decentralized FL training round execution and model export."""
        progress_updates = []

        def callback(pct, msg):
            progress_updates.append((pct, msg))

        res = FLConsortiumService.trigger_consortium_round(
            strategy="FedAvg",
            epochs=1,
            dp_epsilon=3.0,
            progress_callback=callback
        )

        self.assertTrue(res["success"])
        self.assertGreater(res["round_number"], 0)
        self.assertEqual(res["branches_count"], 4)
        self.assertGreater(res["train_loss"], 0.0)
        self.assertGreater(res["gini_score"], 0.0)
        full_ckpt_path = os.path.join(PROJECT_ROOT, res["checkpoint_path"])
        self.assertTrue(os.path.exists(full_ckpt_path))
        self.assertGreater(len(progress_updates), 3)

        # Cleanup generated test checkpoint
        if os.path.exists(full_ckpt_path) and "fl_consortium_round_" in full_ckpt_path:
            try:
                os.remove(full_ckpt_path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()

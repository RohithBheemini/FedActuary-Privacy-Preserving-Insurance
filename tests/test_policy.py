"""
Unit tests for FedActuary Policy Management, Claims Analytics, and Endorsements.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

from src.data import (load_severity_data, aggregate_severity_by_policy,
                      load_combined_policy_dataset, load_preprocessing_artifacts)
from src.model import MultipleRegression
from src.policy import PolicyManager, BILLING_FREQUENCIES


class TestPolicyManagement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_store = os.path.join(cls.temp_dir, "test_store.json")
        
        # Initialize PolicyManager with small row count for rapid testing
        cls.pm = PolicyManager(store_path=cls.temp_store, n_rows=5000)

        # Load checkpoints if available
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        model_path = os.path.join(ckpt_dir, "federated.pt")
        if os.path.exists(model_path):
            cls.scaler, cls.feature_names = load_preprocessing_artifacts(ckpt_dir)
            cls.model = MultipleRegression(num_features=len(cls.feature_names))
            cls.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            cls.model.eval()
        else:
            cls.model = None
            cls.scaler = None
            cls.feature_names = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_severity_dataset_loading(self):
        """Test freMTPL2sev loading and integrity."""
        df_sev = load_severity_data()
        self.assertGreater(len(df_sev), 20000)
        self.assertIn("IDpol", df_sev.columns)
        self.assertIn("ClaimAmount", df_sev.columns)
        self.assertTrue((df_sev["ClaimAmount"] >= 0).all())

    def test_combined_dataset_columns(self):
        """Test merged policy dataset features."""
        df = self.pm.get_all_policies()
        expected_cols = [
            "IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge",
            "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region",
            "SevClaimCount", "TotalClaimAmount", "AvgClaimAmount", "MaxClaimAmount",
            "BaseCoverageLimit", "TopUpAmount", "TotalCoverageLimit",
            "RemainingClaimAmount", "ClaimUtilizationPct", "PolicyFrequency",
            "AnnualPremium", "PolicyStatus"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_portfolio_total_claims_generated(self):
        """Test portfolio stats answering total claims generated."""
        stats = self.pm.get_portfolio_stats()
        self.assertGreater(stats["total_policies"], 0)
        self.assertGreaterEqual(stats["total_claims_generated"], 0)
        self.assertGreaterEqual(stats["total_claim_amount_generated"], 0.0)
        self.assertIn("claims_by_region", stats)
        self.assertIn("claims_by_age", stats)
        self.assertIn("claim_distribution", stats)

    def test_policy_query_pagination(self):
        """Test querying all policies as a dataset with filters and pagination."""
        page_df, total = self.pm.query_policies(page=1, page_size=20)
        self.assertEqual(len(page_df), 20)
        self.assertEqual(total, len(self.pm.get_all_policies()))

        # Filter by region
        reg_df, reg_total = self.pm.query_policies(region="R11", page=1, page_size=10)
        self.assertTrue((reg_df["Region"] == "R11").all())

    def test_remaining_claim_calculation(self):
        """Test remaining claim calculation for a user."""
        df = self.pm.get_all_policies()
        sample_id = int(df.iloc[0]["IDpol"])
        rem_info = self.pm.get_claim_remaining(sample_id)

        self.assertEqual(rem_info["id_pol"], sample_id)
        self.assertGreater(rem_info["base_coverage_limit"], 0)
        self.assertEqual(
            rem_info["total_coverage_limit"],
            rem_info["base_coverage_limit"] + rem_info["top_up_amount"]
        )
        self.assertAlmostEqual(
            rem_info["remaining_claim_amount"],
            max(0.0, rem_info["total_coverage_limit"] - rem_info["total_claim_incurred"]),
            places=1
        )
        self.assertIn(rem_info["status_code"], ["INTACT", "HEALTHY", "MODERATE", "CRITICAL", "EXHAUSTED"])

    def test_top_up_eligibility_and_execution(self):
        """Test checking top-up eligibility and executing top-up."""
        df = self.pm.get_all_policies()
        sample_id = int(df.iloc[1]["IDpol"])

        elig = self.pm.check_top_up_eligibility(sample_id)
        self.assertTrue(elig["eligible"])
        self.assertGreater(len(elig["quotes"]), 0)

        # Apply €10,000 top up
        before_claim = self.pm.get_claim_remaining(sample_id)
        res = self.pm.apply_top_up(sample_id, 10000.0, notes="Unit test top-up")

        self.assertTrue(res["success"])
        self.assertEqual(res["top_up_amount"], 10000.0)
        self.assertEqual(res["new_total_limit"], before_claim["total_coverage_limit"] + 10000.0)

        # Verify claim remaining updated
        after_claim = self.pm.get_claim_remaining(sample_id)
        self.assertEqual(after_claim["top_up_amount"], before_claim["top_up_amount"] + 10000.0)
        self.assertEqual(after_claim["remaining_claim_amount"], before_claim["remaining_claim_amount"] + 10000.0)

    def test_update_policy_details_and_rescore(self):
        """Test editing policy details and dynamic re-scoring."""
        df = self.pm.get_all_policies()
        sample_id = int(df.iloc[2]["IDpol"])

        updates = {"DrivAge": 35, "VehAge": 5, "VehPower": 8}
        res = self.pm.update_policy_details(
            sample_id,
            updates=updates,
            model=self.model,
            scaler=self.scaler,
            feature_names=self.feature_names
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["diff"]["DrivAge"]["after"], 35)
        self.assertEqual(res["diff"]["VehAge"]["after"], 5)
        self.assertEqual(res["diff"]["VehPower"]["after"], 8)

        # Verify updated policy reflects changes
        updated_pol = self.pm.get_policy(sample_id)
        self.assertEqual(updated_pol["DrivAge"], 35)

    def test_change_policy_frequency(self):
        """Test changing policy frequency and installment schedule."""
        df = self.pm.get_all_policies()
        sample_id = int(df.iloc[3]["IDpol"])

        # Switch to Monthly
        res = self.pm.update_policy_frequency(sample_id, "Monthly")
        self.assertTrue(res["success"])
        self.assertEqual(res["new_frequency"], "Monthly")
        self.assertEqual(res["installments_count"], 12)
        self.assertAlmostEqual(res["term_exposure"], BILLING_FREQUENCIES["Monthly"]["term_exposure"], places=2)

        # Switch to Semi-Annual
        res2 = self.pm.update_policy_frequency(sample_id, "Semi-Annual")
        self.assertTrue(res2["success"])
        self.assertEqual(res2["new_frequency"], "Semi-Annual")
        self.assertEqual(res2["installments_count"], 2)


if __name__ == "__main__":
    unittest.main()

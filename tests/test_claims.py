"""
Unit tests for FedSure Mutual Claims & FNOL Service.
"""

import os
import unittest
from src.claims import ClaimsService
from src.database import init_db, get_policy_from_db, upsert_policy_in_db


class TestClaimsService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        # Seed a test policy
        upsert_policy_in_db({
            "id_pol": 99901,
            "holder_name": "Test Claimant",
            "holder_email": "claimant@test.com",
            "driv_age": 40,
            "bonus_malus": 50,
            "exposure": 1.0,
            "veh_power": 6,
            "veh_age": 3,
            "veh_gas": "Regular",
            "veh_brand": "B12",
            "area": "C",
            "region": "R82",
            "density": 800.0,
            "base_limit": 50000.0,
            "top_up_amount": 0.0,
            "total_limit": 50000.0,
            "frequency": "Annual",
            "annual_premium": 250.0,
            "status": "Active (Clean)",
            "fl_predicted_claim_freq": 0.07,
            "fl_risk_tier": "Low Risk",
            "version": 1
        })

    def test_file_fnol_valid(self):
        """Test valid FNOL claim submission with automatic AI risk triage."""
        res = ClaimsService.file_fnol(
            id_pol=99901,
            claimant_name="Test Claimant",
            incident_date="2026-10-01",
            claim_type="Windshield Damage",
            amount_claimed=650.0,
            description="Minor rock chip on highway",
            fl_predicted_freq=0.06
        )
        self.assertTrue(res["success"])
        self.assertIn("CLM-", res["claim_id"])
        self.assertEqual(res["id_pol"], 99901)
        self.assertEqual(res["amount_claimed"], 650.0)
        self.assertGreaterEqual(res["fl_fraud_risk_score"], 0.0)
        self.assertLessEqual(res["fl_fraud_risk_score"], 1.0)
        self.assertIn("Auto Fast-Track", res["triage_recommendation"])

    def test_file_fnol_zero_amount(self):
        """Test that invalid claim amount <= 0 is rejected."""
        res = ClaimsService.file_fnol(
            id_pol=99901,
            claimant_name="Test Claimant",
            incident_date="2026-10-01",
            claim_type="Dent",
            amount_claimed=0.0,
            description="No damage"
        )
        self.assertFalse(res["success"])

    def test_claim_retrieval_and_adjudication(self):
        """Test retrieving filed claims and adjudicating an outcome."""
        claim_res = ClaimsService.file_fnol(
            id_pol=99901,
            claimant_name="Test Claimant",
            incident_date="2026-10-02",
            claim_type="Collision",
            amount_claimed=1200.0,
            description="Parking lot collision"
        )
        claim_id = claim_res["claim_id"]

        # Retrieve claims for this policy
        claims = ClaimsService.get_policy_claims(99901)
        self.assertTrue(any(c["claim_id"] == claim_id for c in claims))

        # Adjudicate
        adj = ClaimsService.adjudicate(
            claim_id=claim_id,
            decision="Approved",
            payout_amount=1200.0,
            underwriter_notes="Reviewed and confirmed damage estimate."
        )
        self.assertTrue(adj["success"])
        self.assertEqual(adj["decision"], "Approved")
        self.assertEqual(adj["payout_amount"], 1200.0)

    def test_high_risk_siu_flag(self):
        """Test that high claim amount exceeding limit or suspicious flags triggers SIU."""
        res = ClaimsService.file_fnol(
            id_pol=99901,
            claimant_name="Test Claimant",
            incident_date="2026-10-03",
            claim_type="Hit and run",
            amount_claimed=75000.0,  # Exceeds 50k limit
            description="Suspicious collision in unmonitored area",
            fl_predicted_freq=0.45
        )
        self.assertTrue(res["success"])
        self.assertIn("High-Risk SIU Flag", res["triage_recommendation"])


if __name__ == "__main__":
    unittest.main()

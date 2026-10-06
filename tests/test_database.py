"""
Unit tests for FedSure Mutual SQLite Persistence Layer.
"""

import json
import unittest
from src.database import (
    init_db, get_all_users, get_user_by_username,
    upsert_policy_in_db, get_policy_from_db,
    add_endorsement, get_endorsements_for_policy, update_endorsement_status,
    add_top_up_tx, get_top_up_history,
    log_fl_round, get_fl_consortium_rounds, set_production_checkpoint
)


class TestDatabaseLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_users_seeded_with_rbac(self):
        """Verify initial users exist with distinct role permissions."""
        users = get_all_users()
        self.assertGreaterEqual(len(users), 4)
        roles = {u["role"] for u in users}
        self.assertIn("policyholder", roles)
        self.assertIn("underwriter", roles)
        self.assertIn("agent", roles)
        self.assertIn("admin", roles)

        user = get_user_by_username("alice")
        self.assertIsNotNone(user)
        self.assertEqual(user["role"], "policyholder")

    def test_policy_upsert_and_fetch(self):
        """Test persisting and querying a policy in SQLite."""
        pol_data = {
            "id_pol": 88801,
            "holder_name": "DB Test User",
            "holder_email": "test@fedsure.eu",
            "driv_age": 42,
            "bonus_malus": 55,
            "exposure": 0.8,
            "veh_power": 7,
            "veh_age": 2,
            "veh_gas": "Regular",
            "veh_brand": "B10",
            "area": "D",
            "region": "R24",
            "density": 1200.0,
            "base_limit": 60000.0,
            "top_up_amount": 0.0,
            "total_limit": 60000.0,
            "frequency": "Quarterly",
            "annual_premium": 320.0,
            "status": "Active (Clean)",
            "fl_predicted_claim_freq": 0.082,
            "fl_risk_tier": "Standard Risk",
            "version": 1
        }
        upsert_policy_in_db(pol_data)

        pol = get_policy_from_db(88801)
        self.assertIsNotNone(pol)
        self.assertEqual(pol["holder_name"], "DB Test User")
        self.assertEqual(pol["driv_age"], 42)
        self.assertEqual(pol["frequency"], "Quarterly")

    def test_endorsement_lifecycle(self):
        """Test logging an endorsement request and updating review status."""
        changes = {"VehPower": 8, "VehAge": 3}
        end_id = add_endorsement(
            id_pol=88801,
            requested_by="alice",
            endorsement_type="Vehicle Update",
            changes_json=json.dumps(changes),
            pred_before=0.082,
            pred_after=0.085,
            premium_delta=15.0,
            notes="Upgraded vehicle power"
        )
        self.assertIsNotNone(end_id)
        self.assertIn("END-", end_id)

        ends = get_endorsements_for_policy(88801)
        self.assertTrue(any(e["endorsement_id"] == end_id for e in ends))

        # Update status
        update_endorsement_status(
            endorsement_id=end_id,
            new_status="Approved",
            reviewed_by="sarah_underwriter",
            notes="Verified registration document."
        )

    def test_top_up_history_persistence(self):
        """Test recording and retrieving top-up transactions."""
        tx_id = add_top_up_tx(
            id_pol=88801,
            amount=15000.0,
            premium_charge=30.0,
            new_total_limit=75000.0,
            new_remaining_claim=75000.0,
            payment_method="Credit Card"
        )
        self.assertIn("TOP-", tx_id)

        history = get_top_up_history(88801)
        self.assertTrue(any(h["tx_id"] == tx_id for h in history))

    def test_fl_consortium_round_and_checkpoint_governance(self):
        """Test recording FL consortium round metrics and promoting checkpoints."""
        round_id = log_fl_round(
            consortium_name="PEFIAC",
            round_number=999,
            participating_branches="Branch 1, Branch 2",
            aggregation_strategy="FedAvg",
            dp_epsilon=3.0,
            train_loss=0.215,
            pde_score=5.2,
            gini_score=0.284,
            model_checkpoint_path="checkpoints/test_checkpoint_999.pt",
            is_production=False
        )
        self.assertIsNotNone(round_id)

        # Promote to production
        promote_res = set_production_checkpoint("checkpoints/test_checkpoint_999.pt")
        self.assertTrue(promote_res)

        rounds = get_fl_consortium_rounds()
        prod_round = next((r for r in rounds if r["model_checkpoint_path"] == "checkpoints/test_checkpoint_999.pt"), None)
        self.assertIsNotNone(prod_round)
        self.assertEqual(prod_round["is_production"], 1)


if __name__ == "__main__":
    unittest.main()

"""
Claims & FNOL (First Notice of Loss) Service for FedSure Enterprise Platform.
Handles digital claim filing, automated Federated AI risk scoring & triage,
and underwriter claim adjudication.
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.database import (
    add_claim, get_claims_for_policy, get_all_claims,
    update_claim_adjudication, get_policy_from_db, upsert_policy_in_db,
    get_db_connection
)


class ClaimsService:
    """Manages insurance claim lifecycles, FNOL intake, and automated FL risk assessment."""

    @staticmethod
    def file_fnol(
        id_pol: int,
        claimant_name: str,
        incident_date: str,
        claim_type: str,
        amount_claimed: float,
        description: str,
        policy_details: Optional[Dict[str, Any]] = None,
        fl_predicted_freq: Optional[float] = None,
        evidence_attachment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Files a First Notice of Loss (FNOL).
        Performs automated triage using policy parameters, remaining claim balance,
        and federated learning risk scores.
        """
        amount_claimed = float(amount_claimed)
        if amount_claimed <= 0:
            return {"success": False, "message": "Claim amount must be greater than €0."}

        # Retrieve policy info to verify coverage limit
        remaining_balance = 50000.0
        if policy_details:
            total_limit = float(policy_details.get("TotalCoverageLimit", policy_details.get("total_limit", 50000.0)))
            total_claimed = float(policy_details.get("TotalClaimAmount", 0.0))
            remaining_balance = max(0.0, total_limit - total_claimed)
        else:
            db_pol = get_policy_from_db(id_pol)
            if db_pol:
                total_limit = float(db_pol["total_limit"])
                remaining_balance = total_limit

        # Automated FL Risk Scoring & Triage
        # Base probability calculation based on model prediction and claim amount
        pred_freq = fl_predicted_freq or (float(policy_details.get("PredictedClaimCount", 0.08)) if policy_details else 0.08)
        
        # Risk factors
        amount_ratio = amount_claimed / max(remaining_balance, 1.0)
        
        # Fraud/anomaly risk score between 0.0 and 1.0
        raw_risk = (pred_freq * 2.0) + (0.35 if amount_ratio > 0.5 else 0.1) + (0.25 if "Suspicious" in description or "Hit and run" in claim_type else 0.0)
        fraud_risk_score = round(float(np.clip(raw_risk, 0.05, 0.95)), 2)

        # Triage determination
        if fraud_risk_score < 0.25 and amount_claimed <= 1500.0 and amount_claimed <= remaining_balance:
            triage_rec = "Auto Fast-Track (Low Risk, Instant Payout Eligible)"
        elif fraud_risk_score >= 0.60 or amount_claimed > remaining_balance:
            triage_rec = "High-Risk SIU Flag (Manual Investigation Required)"
        else:
            triage_rec = "Standard Underwriter Review Required"

        if policy_details:
            try:
                upsert_policy_in_db({
                    "id_pol": id_pol,
                    "holder_name": claimant_name or policy_details.get("holder_name"),
                    "driv_age": policy_details.get("DrivAge", policy_details.get("driv_age", 35)),
                    "bonus_malus": policy_details.get("BonusMalus", policy_details.get("bonus_malus", 50)),
                    "exposure": policy_details.get("Exposure", policy_details.get("exposure", 1.0)),
                    "veh_power": policy_details.get("VehPower", policy_details.get("veh_power", 6)),
                    "veh_age": policy_details.get("VehAge", policy_details.get("veh_age", 4)),
                    "veh_gas": policy_details.get("VehGas", policy_details.get("veh_gas", "Regular")),
                    "veh_brand": policy_details.get("VehBrand", policy_details.get("veh_brand", "B12")),
                    "area": policy_details.get("Area", policy_details.get("area", "C")),
                    "region": policy_details.get("Region", policy_details.get("region", "R82")),
                    "density": policy_details.get("Density", policy_details.get("density", 1000.0)),
                    "base_limit": policy_details.get("BaseCoverageLimit", policy_details.get("base_limit", 50000.0)),
                    "top_up_amount": policy_details.get("TopUpAmount", policy_details.get("top_up_amount", 0.0)),
                    "total_limit": policy_details.get("TotalCoverageLimit", policy_details.get("total_limit", 50000.0)),
                    "frequency": policy_details.get("PolicyFrequency", policy_details.get("frequency", "Annual")),
                    "annual_premium": policy_details.get("AnnualPremium", policy_details.get("annual_premium", 250.0)),
                    "status": "Active with Claims",
                    "fl_predicted_claim_freq": pred_freq,
                })
            except Exception:
                pass

        # Record in DB
        claim_id = add_claim(
            id_pol=id_pol,
            claimant_name=claimant_name,
            incident_date=incident_date,
            claim_type=claim_type,
            amount_claimed=amount_claimed,
            description=description,
            fl_fraud_risk_score=fraud_risk_score,
            triage_recommendation=triage_rec,
            evidence_attachment=evidence_attachment
        )

        return {
            "success": True,
            "claim_id": claim_id,
            "id_pol": id_pol,
            "amount_claimed": amount_claimed,
            "remaining_coverage": remaining_balance,
            "fl_fraud_risk_score": fraud_risk_score,
            "triage_recommendation": triage_rec,
            "message": f"Claim {claim_id} successfully logged and triaged."
        }

    @staticmethod
    def get_policy_claims(id_pol: int) -> List[Dict[str, Any]]:
        return get_claims_for_policy(id_pol)

    @staticmethod
    def list_all_claims(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        return get_all_claims(status_filter)

    @staticmethod
    def adjudicate(
        claim_id: str,
        decision: str,  # 'Approved', 'Denied', 'Under Review', 'Settled'
        payout_amount: float,
        underwriter_notes: str,
        policy_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Adjudicates a claim. When payout is approved or settled, updates the claim status
        and optionally recalculates policy limits.
        """
        conn = get_db_connection()
        claim_row = conn.execute("SELECT * FROM claims WHERE claim_id = ?;", (claim_id,)).fetchone()
        conn.close()

        if not claim_row:
            return {"success": False, "message": f"Claim {claim_id} not found."}

        claim_dict = dict(claim_row)
        id_pol = claim_dict["id_pol"]

        update_claim_adjudication(
            claim_id=claim_id,
            new_status=decision,
            approved_payout=float(payout_amount),
            underwriter_notes=underwriter_notes
        )

        # If payout occurred, update policy in SQLite
        if decision in ("Approved", "Settled") and payout_amount > 0:
            db_pol = get_policy_from_db(id_pol)
            if db_pol:
                new_status = "Active with Claims"
                upsert_policy_in_db({
                    "id_pol": id_pol,
                    "status": new_status
                })

        return {
            "success": True,
            "claim_id": claim_id,
            "decision": decision,
            "payout_amount": payout_amount,
            "message": f"Claim {claim_id} adjudication saved ({decision})."
        }

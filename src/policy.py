"""
Policy Management & Actuarial Portfolio Service for FedActuary.

Provides end-to-end management for insurance policies:
1. Portfolio Analytics: Total claims generated (counts, payouts, distributions across regions/age/brand).
2. Policy Dataset Management: Merged 678k policy records with fast search, filters, pagination, and export.
3. Remaining Claim Tracking: Calculation of coverage limits, incurred claims, remaining claim balance, and status.
4. Policy Top-Up Engine: Eligibility assessment, additional premium quoting, and top-up execution.
5. Policy Modification & Frequency Management: Live endorsement of policy features, billing frequency changes
   (Annual, Semi-Annual, Quarterly, Monthly), and real-time FedActuary neural net re-scoring.
"""

import copy
import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.data import (AREA_MAP, REGIONS, SCALE_COLS, VEH_BRANDS,
                      aggregate_severity_by_policy,
                      load_combined_policy_dataset, load_preprocessing_artifacts,
                      load_raw_features, load_severity_data,
                      preprocess_single_record)
from src.model import MultipleRegression
from src.train import predict
from src.database import (
    init_db, upsert_policy_in_db, add_endorsement, add_top_up_tx,
    get_all_policies_from_db, get_policy_from_db
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
STORE_PATH = os.path.join(DATA_DIR, "policy_store.json")

# Standard Billing Frequencies
BILLING_FREQUENCIES = {
    "Annual": {
        "installments_per_year": 1,
        "term_exposure": 1.0,
        "installment_factor": 1.0,
        "description": "1 single annual payment (No surcharge)",
    },
    "Semi-Annual": {
        "installments_per_year": 2,
        "term_exposure": 0.5,
        "installment_factor": 0.51,  # 2% admin loading
        "description": "2 half-yearly payments (2% processing factor)",
    },
    "Quarterly": {
        "installments_per_year": 4,
        "term_exposure": 0.25,
        "installment_factor": 0.26,  # 4% admin loading
        "description": "4 quarterly payments (4% processing factor)",
    },
    "Monthly": {
        "installments_per_year": 12,
        "term_exposure": 0.0833,
        "installment_factor": 0.0883,  # 6% admin loading
        "description": "12 monthly installments (6% processing factor)",
    },
}


class PolicyManager:
    """
    Central service for policy lookups, claims tracking, top-ups, endorsements,
    and portfolio-wide actuarial analytics.
    """

    def __init__(
        self,
        freq_path: Optional[str] = None,
        sev_path: Optional[str] = None,
        store_path: Optional[str] = None,
        n_rows: Optional[int] = None,
    ):
        self.freq_path = freq_path or os.path.join(DATA_DIR, "freMTPL2freq.csv")
        self.sev_path = sev_path or os.path.join(DATA_DIR, "freMTPL2sev.csv")
        self.store_path = store_path or STORE_PATH
        self.n_rows = n_rows

        self._raw_sev: Optional[pd.DataFrame] = None
        self._df_portfolio: Optional[pd.DataFrame] = None
        self._overrides: Dict[str, Dict[str, Any]] = {}

        self._load_store()
        try:
            init_db()
        except Exception:
            pass
        self._initialize_data()

    def _load_store(self):
        """Loads persistent policy endorsements and top-ups."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self._overrides = json.load(f)
            except Exception:
                self._overrides = {}
        else:
            self._overrides = {}

    def _save_store(self):
        """Persists policy modifications to disk."""
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self._overrides, f, indent=2)

    def _initialize_data(self):
        """Loads and precomputes merged portfolio dataframe with active overrides."""
        if os.path.exists(self.sev_path):
            self._raw_sev = load_severity_data(self.sev_path)
        else:
            self._raw_sev = pd.DataFrame(columns=["IDpol", "ClaimAmount"])

        self._df_portfolio = load_combined_policy_dataset(
            freq_path=self.freq_path,
            sev_path=self.sev_path,
            n_rows=self.n_rows,
        )

        # Apply persisted overrides
        self._apply_all_overrides()

    def _apply_all_overrides(self):
        """Applies persistent top-ups and endorsements into dataframe."""
        if not self._overrides or self._df_portfolio is None:
            return

        for id_str, override in self._overrides.items():
            try:
                id_pol = int(id_str)
            except ValueError:
                continue

            mask = self._df_portfolio["IDpol"] == id_pol
            if not mask.any():
                continue

            row_idx = self._df_portfolio.index[mask][0]

            # Apply top-ups
            top_up_total = sum(t.get("amount", 0.0) for t in override.get("top_ups", []))
            base_limit = override.get("base_limit", self._df_portfolio.at[row_idx, "BaseCoverageLimit"])
            self._df_portfolio.at[row_idx, "BaseCoverageLimit"] = float(base_limit)
            self._df_portfolio.at[row_idx, "TopUpAmount"] = float(top_up_total)
            total_limit = base_limit + top_up_total
            self._df_portfolio.at[row_idx, "TotalCoverageLimit"] = float(total_limit)

            incurred = float(self._df_portfolio.at[row_idx, "TotalClaimAmount"])
            remaining = max(0.0, total_limit - incurred)
            self._df_portfolio.at[row_idx, "RemainingClaimAmount"] = round(remaining, 2)
            self._df_portfolio.at[row_idx, "ClaimUtilizationPct"] = round(
                min(100.0, (incurred / total_limit * 100.0) if total_limit > 0 else 100.0), 2
            )

            # Apply policy details
            details = override.get("details", {})
            for k, v in details.items():
                if k in self._df_portfolio.columns:
                    self._df_portfolio.at[row_idx, k] = v

            # Apply frequency
            if "frequency" in override:
                self._df_portfolio.at[row_idx, "PolicyFrequency"] = override["frequency"]

            # Update status
            rem = self._df_portfolio.at[row_idx, "RemainingClaimAmount"]
            util = self._df_portfolio.at[row_idx, "ClaimUtilizationPct"]
            if rem <= 0:
                self._df_portfolio.at[row_idx, "PolicyStatus"] = "Limit Exhausted"
            elif util >= 80.0:
                self._df_portfolio.at[row_idx, "PolicyStatus"] = "Critical (Near Limit)"
            elif util > 0:
                self._df_portfolio.at[row_idx, "PolicyStatus"] = "Active with Claims"
            else:
                self._df_portfolio.at[row_idx, "PolicyStatus"] = "Active (Clean)"

        # Also sync any records modified in SQLite enterprise database
        try:
            db_policies = get_all_policies_from_db()
            for p in db_policies:
                id_pol = p["id_pol"]
                mask = self._df_portfolio["IDpol"] == id_pol
                if mask.any():
                    row_idx = self._df_portfolio.index[mask][0]
                    for col_map, val in [
                        ("DrivAge", p["driv_age"]), ("BonusMalus", p["bonus_malus"]),
                        ("VehPower", p["veh_power"]), ("VehAge", p["veh_age"]),
                        ("VehGas", p["veh_gas"]), ("VehBrand", p["veh_brand"]),
                        ("Area", p["area"]), ("Region", p["region"]),
                        ("Density", p["density"]), ("BaseCoverageLimit", p["base_limit"]),
                        ("PolicyFrequency", p["frequency"]), ("AnnualPremium", p["annual_premium"]),
                        ("PolicyStatus", p["status"])
                    ]:
                        if col_map in self._df_portfolio.columns and val is not None:
                            self._df_portfolio.at[row_idx, col_map] = val

                    top_up_val = max(float(p["top_up_amount"]), float(self._df_portfolio.at[row_idx, "TopUpAmount"]))
                    tot_lim_val = max(float(p["total_limit"]), float(self._df_portfolio.at[row_idx, "TotalCoverageLimit"]))
                    self._df_portfolio.at[row_idx, "TopUpAmount"] = top_up_val
                    self._df_portfolio.at[row_idx, "TotalCoverageLimit"] = tot_lim_val

                    incurred = float(self._df_portfolio.at[row_idx, "TotalClaimAmount"])
                    rem = max(0.0, tot_lim_val - incurred)
                    self._df_portfolio.at[row_idx, "RemainingClaimAmount"] = round(rem, 2)
                    self._df_portfolio.at[row_idx, "ClaimUtilizationPct"] = round(
                        min(100.0, (incurred / tot_lim_val * 100.0) if tot_lim_val > 0 else 100.0), 2
                    )
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # 1. Total Claims Generated & Portfolio Analytics
    # --------------------------------------------------------------------------
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """
        Answers: 'Total claims generated' across the portfolio.
        Computes aggregate metrics, claim frequencies, severity distributions,
        and geographic breakdowns.
        """
        df = self._df_portfolio
        total_policies = len(df)
        total_claims_gen = int(df["ClaimNb"].sum())
        total_claims_recorded_sev = len(self._raw_sev) if self._raw_sev is not None else 0
        total_claim_amount = float(df["TotalClaimAmount"].sum())

        policies_with_claims = int((df["ClaimNb"] > 0).sum())
        zero_claim_policies = total_policies - policies_with_claims
        total_exposure = float(df["Exposure"].sum())
        overall_frequency = (total_claims_gen / total_exposure) if total_exposure > 0 else 0.0

        avg_severity_per_claim = (
            float(self._raw_sev["ClaimAmount"].mean()) if self._raw_sev is not None and len(self._raw_sev) > 0 else 0.0
        )
        avg_severity_per_claiming_policy = (
            (total_claim_amount / policies_with_claims) if policies_with_claims > 0 else 0.0
        )

        # Claims generated by Region
        claims_by_region = df.groupby("Region", observed=False).agg(
            TotalPolicies=("IDpol", "count"),
            ClaimsGenerated=("ClaimNb", "sum"),
            TotalClaimAmount=("TotalClaimAmount", "sum")
        ).reset_index().to_dict(orient="records")

        # Claims generated by Vehicle Brand
        claims_by_brand = df.groupby("VehBrand", observed=False).agg(
            TotalPolicies=("IDpol", "count"),
            ClaimsGenerated=("ClaimNb", "sum"),
            TotalClaimAmount=("TotalClaimAmount", "sum")
        ).reset_index().to_dict(orient="records")

        # Claims generated by Area
        claims_by_area = df.groupby("Area", observed=False).agg(
            TotalPolicies=("IDpol", "count"),
            ClaimsGenerated=("ClaimNb", "sum"),
            TotalClaimAmount=("TotalClaimAmount", "sum")
        ).reset_index().to_dict(orient="records")

        # Claims generated by Driver Age Group
        age_bins = [17, 25, 35, 50, 65, 100]
        age_labels = ["18-25 (Young)", "26-35 (Early Career)", "36-50 (Experienced)", "51-65 (Mature)", "66+ (Senior)"]
        df_age = df.copy()
        df_age["AgeGroup"] = pd.cut(df_age["DrivAge"], bins=age_bins, labels=age_labels)
        claims_by_age = df_age.groupby("AgeGroup", observed=False).agg(
            TotalPolicies=("IDpol", "count"),
            ClaimsGenerated=("ClaimNb", "sum"),
            TotalClaimAmount=("TotalClaimAmount", "sum")
        ).reset_index().to_dict(orient="records")

        # Claim count distribution breakdown
        claim_dist = df["ClaimNb"].value_counts().sort_index().to_dict()

        return {
            "total_policies": total_policies,
            "total_claims_generated": total_claims_gen,
            "total_severity_claims_recorded": total_claims_recorded_sev,
            "total_claim_amount_generated": round(total_claim_amount, 2),
            "policies_with_claims": policies_with_claims,
            "zero_claim_policies": zero_claim_policies,
            "percent_policies_with_claims": round((policies_with_claims / total_policies) * 100.0, 2),
            "total_exposure_years": round(total_exposure, 2),
            "overall_annual_claim_frequency": round(overall_frequency, 4),
            "avg_severity_per_claim": round(avg_severity_per_claim, 2),
            "avg_severity_per_claiming_policy": round(avg_severity_per_claiming_policy, 2),
            "claims_by_region": claims_by_region,
            "claims_by_brand": claims_by_brand,
            "claims_by_area": claims_by_area,
            "claims_by_age": claims_by_age,
            "claim_distribution": {int(k): int(v) for k, v in claim_dist.items()},
            # Compatibility aliases
            "total_claims_count": total_claims_gen,
            "total_claim_payout": round(total_claim_amount, 2),
            "average_severity": round(avg_severity_per_claim, 2),
            "distribution_by_region": {str(r["Region"]): int(r["TotalPolicies"]) for r in claims_by_region},
            "distribution_by_driver_age_group": {str(r["AgeGroup"]): int(r["TotalPolicies"]) for r in claims_by_age},
        }

    # --------------------------------------------------------------------------
    # 2. Details of All Policies as a Dataset
    # --------------------------------------------------------------------------
    def get_all_policies(self) -> pd.DataFrame:
        """Returns the complete merged policy dataset."""
        return self._df_portfolio

    def query_policies(
        self,
        id_pol: Optional[int] = None,
        region: Optional[str] = None,
        area: Optional[str] = None,
        has_claims: Optional[bool] = None,
        policy_status: Optional[str] = None,
        frequency: Optional[str] = None,
        min_driver_age: Optional[int] = None,
        max_driver_age: Optional[int] = None,
        min_bonus_malus: Optional[int] = None,
        max_bonus_malus: Optional[int] = None,
        page: int = 1,
        page_size: int = 25,
        sort_by: str = "IDpol",
        ascending: bool = True,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Filters and paginates the policy dataset.
        Returns (page_dataframe, total_matches).
        """
        df = self._df_portfolio
        mask = pd.Series(True, index=df.index)

        if id_pol is not None:
            mask = mask & (df["IDpol"] == id_pol)
        if region:
            mask = mask & (df["Region"] == region)
        if area:
            mask = mask & (df["Area"] == area)
        if has_claims is True:
            mask = mask & (df["ClaimNb"] > 0)
        elif has_claims is False:
            mask = mask & (df["ClaimNb"] == 0)
        if policy_status:
            mask = mask & (df["PolicyStatus"] == policy_status)
        if frequency:
            mask = mask & (df["PolicyFrequency"] == frequency)
        if min_driver_age is not None:
            mask = mask & (df["DrivAge"] >= min_driver_age)
        if max_driver_age is not None:
            mask = mask & (df["DrivAge"] <= max_driver_age)
        if min_bonus_malus is not None:
            mask = mask & (df["BonusMalus"] >= min_bonus_malus)
        if max_bonus_malus is not None:
            mask = mask & (df["BonusMalus"] <= max_bonus_malus)

        filtered = df[mask]
        total_matches = len(filtered)

        if sort_by in filtered.columns:
            filtered = filtered.sort_values(by=sort_by, ascending=ascending)

        start = (page - 1) * page_size
        end = start + page_size
        page_df = filtered.iloc[start:end].copy()

        return page_df, total_matches

    def get_sample_policy_ids(self) -> Dict[str, int]:
        """Returns interesting representative sample policy IDs for instant demo."""
        df = self._df_portfolio
        samples = {}

        # 1. Multi-claim policy
        multi = df[df["ClaimNb"] >= 2]
        if not multi.empty:
            samples["Multiple Claims (Active)"] = int(multi.iloc[0]["IDpol"])

        # 2. Single high severity claim
        high_sev = df[df["TotalClaimAmount"] > 5000]
        if not high_sev.empty:
            samples["High Severity Claim (>€5,000)"] = int(high_sev.iloc[0]["IDpol"])

        # 3. Clean safe senior driver
        clean_sr = df[(df["ClaimNb"] == 0) & (df["BonusMalus"] <= 60) & (df["DrivAge"] >= 50)]
        if not clean_sr.empty:
            samples["Clean Safe Senior Driver"] = int(clean_sr.iloc[0]["IDpol"])

        # 4. Young driver with penalty malus
        young_risk = df[(df["DrivAge"] <= 25) & (df["BonusMalus"] >= 100)]
        if not young_risk.empty:
            samples["Young Driver with Malus"] = int(young_risk.iloc[0]["IDpol"])

        return samples

    def get_policy(self, id_pol: int) -> Optional[Dict[str, Any]]:
        """Retrieves complete profile details for a specific policy ID."""
        df = self._df_portfolio
        match = df[df["IDpol"] == id_pol]
        if match.empty:
            return None

        record = match.iloc[0].to_dict()

        # Retrieve specific claims line items from raw severity
        claims_list = []
        if self._raw_sev is not None:
            sev_records = self._raw_sev[self._raw_sev["IDpol"] == id_pol]
            for idx, r in enumerate(sev_records.itertuples(), start=1):
                claims_list.append({
                    "claim_index": idx,
                    "claim_amount": float(r.ClaimAmount),
                })

        record["claims_list"] = claims_list

        # Attach override histories if present
        override = self._overrides.get(str(id_pol), {})
        record["top_ups_history"] = override.get("top_ups", [])
        record["endorsements_history"] = override.get("endorsements", [])

        return record

    # --------------------------------------------------------------------------
    # 3. How Much Claim is Remaining for a User
    # --------------------------------------------------------------------------
    def get_claim_remaining(self, id_pol: int) -> Dict[str, Any]:
        """
        Answers: 'how much claim is remaining for a user'.
        Calculates base coverage, top-up coverage, total limit, total claims generated,
        and net remaining claim amount.
        """
        policy = self.get_policy(id_pol)
        if policy is None:
            raise ValueError(f"Policy ID {id_pol} not found in portfolio.")

        base_limit = float(policy.get("BaseCoverageLimit", 50000.0))
        top_up_total = float(policy.get("TopUpAmount", 0.0))
        total_limit = base_limit + top_up_total
        total_claims_gen = int(policy.get("ClaimNb", 0))
        incurred_claims = float(policy.get("TotalClaimAmount", 0.0))

        remaining_claim = max(0.0, total_limit - incurred_claims)
        utilization_pct = min(100.0, (incurred_claims / total_limit * 100.0) if total_limit > 0 else 100.0)
        remaining_pct = max(0.0, 100.0 - utilization_pct)

        if remaining_claim <= 0:
            status_desc = "Exhausted (Coverage Limit Fully Consumed)"
            status_code = "EXHAUSTED"
        elif utilization_pct >= 80.0:
            status_desc = "Critical (Over 80% of Limit Consumed - Immediate Top-Up Recommended)"
            status_code = "CRITICAL"
        elif utilization_pct >= 50.0:
            status_desc = "Moderate (50% - 80% of Limit Consumed)"
            status_code = "MODERATE"
        elif utilization_pct > 0.0:
            status_desc = "Healthy (<50% of Limit Consumed)"
            status_code = "HEALTHY"
        else:
            status_desc = "100% Intact (Zero Claims Incurred)"
            status_code = "INTACT"

        return {
            "id_pol": id_pol,
            "base_coverage_limit": round(base_limit, 2),
            "top_up_amount": round(top_up_total),
            "total_coverage_limit": round(total_limit, 2),
            "total_claims_generated": total_claims_gen,
            "total_claim_incurred": round(incurred_claims, 2),
            "remaining_claim_amount": round(remaining_claim, 2),
            "claim_utilization_pct": round(utilization_pct, 2),
            "remaining_claim_pct": round(remaining_pct, 2),
            "status_code": status_code,
            "status_description": status_desc,
            "claims_breakdown": policy.get("claims_list", []),
            "top_ups_history": policy.get("top_ups_history", []),
        }

    # --------------------------------------------------------------------------
    # 4. Can They Top Up Their Policy? (Eligibility & Top-Up Execution)
    # --------------------------------------------------------------------------
    def check_top_up_eligibility(self, id_pol: int) -> Dict[str, Any]:
        """
        Answers: 'can they top up their policy'.
        Evaluates risk parameters, bonus-malus rating, and limits to verify
        top-up feasibility and quote additional premiums.
        """
        policy = self.get_policy(id_pol)
        if policy is None:
            return {"eligible": False, "reason": f"Policy ID {id_pol} not found."}

        bm = float(policy.get("BonusMalus", 100))
        remaining = float(policy.get("RemainingClaimAmount", 50000.0))
        total_limit = float(policy.get("TotalCoverageLimit", 50000.0))
        current_top_ups = float(policy.get("TopUpAmount", 0.0))

        # Actuarial Rules for Top-Up:
        # 1. Total top-up cumulative cap is €100,000 per policy.
        # 2. Risk surcharge applied if BonusMalus > 100.
        max_additional_top_up = max(0.0, 100000.0 - current_top_ups)

        if max_additional_top_up <= 0:
            return {
                "eligible": False,
                "reason": "Policy has already reached the maximum allowable cumulative top-up limit (€100,000).",
                "max_top_up_allowed": 0.0,
                "current_limit": total_limit,
                "remaining_claim": remaining,
            }

        # Pricing formula: Base rate 0.4% of top-up amount * (BonusMalus / 100)
        risk_tier = "Standard"
        risk_multiplier = bm / 100.0
        if bm > 120:
            risk_tier = "High Risk Surcharge"
        elif bm <= 60:
            risk_tier = "Preferred Safe Driver Discount"

        # Pre-calculated quotes for common tiers
        tiers = [5000, 10000, 25000, 50000]
        quotes = []
        for amt in tiers:
            if amt <= max_additional_top_up:
                prem = amt * 0.004 * risk_multiplier
                quotes.append({
                    "top_up_amount": amt,
                    "additional_premium": round(prem, 2),
                    "new_total_limit": round(total_limit + amt, 2),
                    "new_remaining_claim": round(remaining + amt, 2),
                })

        return {
            "eligible": True,
            "reason": "Eligible for coverage enhancement or claim replenishment top-up.",
            "max_top_up_allowed": max_additional_top_up,
            "current_limit": total_limit,
            "current_top_up": current_top_ups,
            "remaining_claim": remaining,
            "risk_tier": risk_tier,
            "risk_multiplier": round(risk_multiplier, 3),
            "quotes": quotes,
        }

    def apply_top_up(
        self,
        id_pol: int,
        top_up_amount: float,
        notes: str = "Standard user requested top-up endorsement"
    ) -> Dict[str, Any]:
        """
        Executes a top-up on the policy:
        Increases policy limit and remaining claim amount, records transaction receipt,
        and saves state persistently.
        """
        eligibility = self.check_top_up_eligibility(id_pol)
        if not eligibility["eligible"]:
            raise ValueError(f"Top-up rejected: {eligibility['reason']}")

        if top_up_amount <= 0:
            raise ValueError("Top-up amount must be strictly positive.")

        if top_up_amount > eligibility["max_top_up_allowed"]:
            raise ValueError(
                f"Top-up amount €{top_up_amount:,.2f} exceeds allowable maximum €{eligibility['max_top_up_allowed']:,.2f}."
            )

        policy = self.get_policy(id_pol)
        bm = float(policy.get("BonusMalus", 100))
        premium_charge = round(top_up_amount * 0.004 * (bm / 100.0), 2)

        tx_id = f"TOP-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{np.random.randint(100, 999)}"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        tx_record = {
            "tx_id": tx_id,
            "timestamp": timestamp,
            "amount": float(top_up_amount),
            "premium_charge": premium_charge,
            "notes": notes,
        }

        id_str = str(id_pol)
        if id_str not in self._overrides:
            self._overrides[id_str] = {"top_ups": [], "endorsements": [], "details": {}}

        self._overrides[id_str].setdefault("top_ups", []).append(tx_record)
        self._save_store()

        all_top_ups = sum(t.get("amount", 0.0) for t in self._overrides[id_str].get("top_ups", []))
        base_lim = float(policy.get("BaseCoverageLimit", 50000.0))
        new_tot = base_lim + all_top_ups
        incurred = float(policy.get("TotalClaimAmount", 0.0))
        new_rem = max(0.0, new_tot - incurred)

        try:
            add_top_up_tx(
                id_pol=id_pol,
                amount=float(top_up_amount),
                premium_charge=float(premium_charge),
                new_total_limit=float(new_tot),
                new_remaining_claim=float(new_rem),
                payment_method="Portal Online Checkout"
            )
        except Exception:
            pass

        self._apply_all_overrides()
        updated_claim = self.get_claim_remaining(id_pol)
        return {
            "success": True,
            "tx_id": tx_id,
            "timestamp": timestamp,
            "id_pol": id_pol,
            "top_up_amount": top_up_amount,
            "premium_charge": premium_charge,
            "prorated_premium": premium_charge,
            "new_total_limit": updated_claim["total_coverage_limit"],
            "new_remaining_claim": updated_claim["remaining_claim_amount"],
            "new_utilization_pct": updated_claim["claim_utilization_pct"],
        }

    # --------------------------------------------------------------------------
    # 5. How We Can Change Policy Details and Policy Frequency
    # --------------------------------------------------------------------------
    def update_policy_details(
        self,
        id_pol: int,
        updates: Dict[str, Any],
        model: Optional[MultipleRegression] = None,
        scaler: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        notes: str = "Policyholder endorsement update",
    ) -> Dict[str, Any]:
        """
        Answers: 'How we can change the policy details'.
        Allows updating driver attributes (DrivAge, BonusMalus), vehicle details
        (VehAge, VehPower, VehBrand, VehGas), and regional parameters (Area, Region, Density).
        Re-evaluates expected claim count with the FedActuary model and calculates premium adjustment.
        """
        policy = self.get_policy(id_pol)
        if policy is None:
            raise ValueError(f"Policy ID {id_pol} not found.")

        valid_fields = [
            "DrivAge", "BonusMalus", "VehAge", "VehPower", "VehBrand",
            "VehGas", "Area", "Region", "Density", "Exposure"
        ]

        old_values = {}
        sanitized_updates = {}
        for k, v in updates.items():
            if k in valid_fields and k in policy:
                old_values[k] = policy[k]
                if k in ["DrivAge", "VehAge", "VehPower", "BonusMalus"]:
                    sanitized_updates[k] = int(v)
                elif k in ["Density", "Exposure"]:
                    sanitized_updates[k] = float(v)
                else:
                    sanitized_updates[k] = str(v)

        if not sanitized_updates:
            return {"success": False, "message": "No valid fields were provided to update."}

        # Calculate FedActuary model predictions before and after if model is available
        pred_before = None
        pred_after = None
        if model is not None and scaler is not None and feature_names is not None:
            rec_before = {k: policy.get(k) for k in valid_fields}
            rec_after = copy.deepcopy(rec_before)
            rec_after.update(sanitized_updates)

            arr_before = preprocess_single_record(rec_before, scaler, feature_names)
            arr_after = preprocess_single_record(rec_after, scaler, feature_names)

            pred_before = float(predict(model, arr_before)[0])
            pred_after = float(predict(model, arr_after)[0])

        tx_id = f"END-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{np.random.randint(100, 999)}"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        endorsement_record = {
            "tx_id": tx_id,
            "timestamp": timestamp,
            "old_values": old_values,
            "new_values": sanitized_updates,
            "notes": notes,
            "pred_before": round(pred_before, 4) if pred_before is not None else None,
            "pred_after": round(pred_after, 4) if pred_after is not None else None,
        }

        id_str = str(id_pol)
        if id_str not in self._overrides:
            self._overrides[id_str] = {"top_ups": [], "endorsements": [], "details": {}}

        self._overrides[id_str].setdefault("endorsements", []).append(endorsement_record)
        self._overrides[id_str].setdefault("details", {}).update(sanitized_updates)

        self._save_store()
        self._apply_all_overrides()

        updated_policy = self.get_policy(id_pol)

        # Persist to SQLite
        try:
            db_fields = {
                "id_pol": id_pol,
                "driv_age": sanitized_updates.get("DrivAge", updated_policy.get("DrivAge")),
                "bonus_malus": sanitized_updates.get("BonusMalus", updated_policy.get("BonusMalus")),
                "veh_power": sanitized_updates.get("VehPower", updated_policy.get("VehPower")),
                "veh_age": sanitized_updates.get("VehAge", updated_policy.get("VehAge")),
                "veh_gas": sanitized_updates.get("VehGas", updated_policy.get("VehGas")),
                "veh_brand": sanitized_updates.get("VehBrand", updated_policy.get("VehBrand")),
                "area": sanitized_updates.get("Area", updated_policy.get("Area")),
                "region": sanitized_updates.get("Region", updated_policy.get("Region")),
                "density": sanitized_updates.get("Density", updated_policy.get("Density")),
                "exposure": sanitized_updates.get("Exposure", updated_policy.get("Exposure")),
                "fl_predicted_claim_freq": pred_after if pred_after is not None else 0.08,
            }
            upsert_policy_in_db(db_fields)
            add_endorsement(
                id_pol=id_pol,
                requested_by="Policyholder Self-Service",
                endorsement_type="Policy Detail Modification",
                changes_json=json.dumps({"old": old_values, "new": sanitized_updates}),
                pred_before=pred_before,
                pred_after=pred_after,
                status="Approved",
                notes=notes
            )
        except Exception:
            pass
        return {
            "success": True,
            "tx_id": tx_id,
            "timestamp": timestamp,
            "id_pol": id_pol,
            "diff": {k: {"before": old_values[k], "after": sanitized_updates[k]} for k in sanitized_updates},
            "pred_before": pred_before,
            "pred_after": pred_after,
            "pred_change_pct": round(((pred_after - pred_before) / pred_before * 100.0), 2) if (pred_before and pred_after) else 0.0,
            "updated_policy": updated_policy,
        }

    def update_policy_frequency(
        self,
        id_pol: int,
        new_frequency: str,
        annual_premium: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Answers: 'How we can change policy frequency'.
        Switches billing frequency between Annual, Semi-Annual, Quarterly, and Monthly.
        Calculates installment schedule, term exposure, and updates persistent store.
        """
        if new_frequency not in BILLING_FREQUENCIES:
            raise ValueError(
                f"Invalid frequency '{new_frequency}'. Must be one of: {list(BILLING_FREQUENCIES.keys())}"
            )

        policy = self.get_policy(id_pol)
        if policy is None:
            raise ValueError(f"Policy ID {id_pol} not found.")

        old_freq = policy.get("PolicyFrequency", "Annual")
        freq_spec = BILLING_FREQUENCIES[new_frequency]

        premium = annual_premium or float(policy.get("AnnualPremium", 250.0))
        installment_amt = round(premium * freq_spec["installment_factor"], 2)
        total_annualized = round(installment_amt * freq_spec["installments_per_year"], 2)

        tx_id = f"FREQ-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{np.random.randint(100, 999)}"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        history_item = {
            "tx_id": tx_id,
            "timestamp": timestamp,
            "old_frequency": old_freq,
            "new_frequency": new_frequency,
            "installment_amount": installment_amt,
            "installments_per_year": freq_spec["installments_per_year"],
            "total_annualized": total_annualized,
        }

        id_str = str(id_pol)
        if id_str not in self._overrides:
            self._overrides[id_str] = {"top_ups": [], "endorsements": [], "details": {}}

        self._overrides[id_str]["frequency"] = new_frequency
        self._overrides[id_str].setdefault("frequency_history", []).append(history_item)

        self._save_store()
        self._apply_all_overrides()

        # Persist to SQLite
        try:
            upsert_policy_in_db({
                "id_pol": id_pol,
                "frequency": new_frequency,
                "annual_premium": total_annualized,
            })
            add_endorsement(
                id_pol=id_pol,
                requested_by="Policyholder Self-Service",
                endorsement_type="Billing Schedule Adjustment",
                changes_json=json.dumps({"old_frequency": old_freq, "new_frequency": new_frequency, "installment": installment_amt}),
                status="Auto-Approved",
                notes=f"Changed frequency to {new_frequency} ({freq_spec['installments_per_year']} installments/year)"
            )
        except Exception:
            pass

        return {
            "success": True,
            "tx_id": tx_id,
            "id_pol": id_pol,
            "old_frequency": old_freq,
            "new_frequency": new_frequency,
            "installment_amount": installment_amt,
            "installments_count": freq_spec["installments_per_year"],
            "total_annualized": total_annualized,
            "term_exposure": freq_spec["term_exposure"],
            "description": freq_spec["description"],
        }

    def get_endorsements(self, id_pol: int) -> List[Dict[str, Any]]:
        """Returns endorsement history for a policy from SQLite and in-memory store."""
        from src.database import get_endorsements_for_policy
        try:
            db_ends = get_endorsements_for_policy(id_pol)
            if db_ends:
                return db_ends
        except Exception:
            pass
        return self._overrides.get(str(id_pol), {}).get("endorsements", [])


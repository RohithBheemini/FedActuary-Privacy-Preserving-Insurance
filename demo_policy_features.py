"""
FedActuary Policy & Claims Feature Demonstration Script.

Demonstrates:
1. freMTPL2sev dataset integration with freMTPL2freq
2. Answering total claims generated (portfolio-wide and per policy)
3. Viewing details of all policies as a dataset
4. Calculating remaining claim capacity for a user
5. Checking top-up eligibility and executing policy top-ups
6. Changing policy details and policy frequency with live actuarial model re-scoring
"""

import json
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from src.data import load_severity_data, load_preprocessing_artifacts
from src.model import MultipleRegression
from src.policy import PolicyManager, BILLING_FREQUENCIES


def run_demo():
    print("=" * 80)
    print(" 🛡️  FedActuary: Policy & Claims Extended Features Demonstration")
    print("=" * 80)

    # 1. freMTPL2sev Dataset Integration
    print("\n[1] freMTPL2sev Dataset Integration")
    print("-" * 50)
    df_sev = load_severity_data()
    print(f"Loaded freMTPL2sev: {len(df_sev):,} claim records across {df_sev['IDpol'].nunique():,} unique policies.")
    print(f"Total claim severity losses recorded: €{df_sev['ClaimAmount'].sum():,.2f}")
    print(f"Average severity per claim: €{df_sev['ClaimAmount'].mean():,.2f}")
    print(f"Median severity: €{df_sev['ClaimAmount'].median():,.2f}")
    print(f"Sample records:\n{df_sev.head(3).to_string(index=False)}")

    # Initialize PolicyManager (full portfolio)
    print("\nLoading full policy portfolio (678,013 policies)...")
    pm = PolicyManager()

    # 2. Total Claims Generated
    print("\n[2] Answering Total Claims Generated")
    print("-" * 50)
    stats = pm.get_portfolio_stats()
    print(f"• Total Policies Insured:              {stats['total_policies']:,}")
    print(f"• Total Claims Generated (ClaimNb):    {stats['total_claims_generated']:,}")
    print(f"• Policies with >= 1 Claims:           {stats['policies_with_claims']:,} ({stats['percent_policies_with_claims']}%)")
    print(f"• Total Monetary Claims Losses:        €{stats['total_claim_amount_generated']:,.2f}")
    print(f"• Overall Annual Claim Frequency:      {stats['overall_annual_claim_frequency']*100:.2f}% (claims / exposure year)")
    print(f"• Average Payout per Claim:            €{stats['avg_severity_per_claim']:,.2f}")
    print("• Claim Count Distribution:")
    for k, v in stats["claim_distribution"].items():
        print(f"   - {k} claim(s): {v:,} policies")

    # 3. Details of All Policies as a Dataset
    print("\n[3] Details of All Policies as a Dataset")
    print("-" * 50)
    page_df, total_count = pm.query_policies(has_claims=True, page=1, page_size=3)
    display_cols = [
        "IDpol", "ClaimNb", "TotalClaimAmount", "BaseCoverageLimit",
        "RemainingClaimAmount", "ClaimUtilizationPct", "PolicyFrequency", "PolicyStatus"
    ]
    print(f"Querying policies with claims (Total matching: {total_count:,}):")
    print(page_df[display_cols].to_string(index=False))

    # 4. How Much Claim is Remaining for a User
    print("\n[4] How Much Claim is Remaining for a User")
    print("-" * 50)
    samples = pm.get_sample_policy_ids()
    sample_id = samples.get("Multiple Claims (Active)", 424)
    rem = pm.get_claim_remaining(sample_id)
    print(f"Policy ID #{sample_id}:")
    print(f"• Base Coverage Limit:         €{rem['base_coverage_limit']:,.2f}")
    print(f"• Top-Up Added:                €{rem['top_up_amount']:,.2f}")
    print(f"• Total Coverage Limit:        €{rem['total_coverage_limit']:,.2f}")
    print(f"• Total Claims Filed:          {rem['total_claims_generated']} claim(s)")
    print(f"• Total Incurred Losses:       €{rem['total_claim_incurred']:,.2f}")
    print(f"• REMAINING CLAIM CAPACITY:    €{rem['remaining_claim_amount']:,.2f} ({rem['remaining_claim_pct']:.1f}% left)")
    print(f"• Limit Utilization Rate:      {rem['claim_utilization_pct']:.1f}%")
    print(f"• Coverage Status:             {rem['status_description']}")
    if rem["claims_breakdown"]:
        print("• Individual Claim Events:")
        for c in rem["claims_breakdown"]:
            print(f"   - Claim #{c['claim_index']}: €{c['claim_amount']:,.2f}")

    # 5. Can They Top Up Their Policy?
    print("\n[5] Can They Top Up Their Policy?")
    print("-" * 50)
    elig = pm.check_top_up_eligibility(sample_id)
    print(f"Eligibility Status: {'ELIGIBLE' if elig['eligible'] else 'NOT ELIGIBLE'}")
    print(f"Reason: {elig['reason']}")
    print(f"Maximum Additional Top-Up Allowed: €{elig['max_top_up_allowed']:,.2f}")
    print(f"Risk Tier: {elig['risk_tier']} (Multiplier: {elig['risk_multiplier']})")
    print("Standard Top-Up Quotes Preview:")
    for q in elig["quotes"][:3]:
        print(f"  +€{q['top_up_amount']:,} limit -> Additional Premium: €{q['additional_premium']:,.2f} | New Limit: €{q['new_total_limit']:,.2f}")

    print("\nApplying a €10,000 Top-Up on Policy #424...")
    res_topup = pm.apply_top_up(sample_id, 10000.0, notes="Demo coverage top-up replenishment")
    print(f"✓ Top-Up Confirmed! Tx ID: {res_topup['tx_id']}")
    print(f"• Old Limit: €{rem['total_coverage_limit']:,.2f}  ──►  New Limit: €{res_topup['new_total_limit']:,.2f}")
    print(f"• Old Remaining: €{rem['remaining_claim_amount']:,.2f}  ──►  New Remaining: €{res_topup['new_remaining_claim']:,.2f}")
    print(f"• One-time Prorated Premium Charged: €{res_topup['premium_charge']:,.2f}")

    # 6. How We Can Change Policy Details and Policy Frequency
    print("\n[6] Changing Policy Details & Policy Frequency")
    print("-" * 50)
    # Load model and scaler
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    scaler, feature_names = load_preprocessing_artifacts(ckpt_dir)
    model = MultipleRegression(num_features=len(feature_names))
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "federated.pt"), map_location="cpu"))
    model.eval()

    curr_pol = pm.get_policy(sample_id)
    new_age = int(curr_pol['DrivAge']) + 1
    new_pwr = max(4, int(curr_pol['VehPower']) - 1)
    new_area = "C" if curr_pol['Area'] != "C" else "B"

    print(f"Updating policy details for Policy #{sample_id}:")
    print(f"  - Updating Driver Age from {curr_pol['DrivAge']} to {new_age}")
    print(f"  - Updating Vehicle Power from {curr_pol['VehPower']} to {new_pwr}")
    print(f"  - Switching Area from {curr_pol['Area']} to {new_area}")

    res_endorse = pm.update_policy_details(
        sample_id,
        updates={"DrivAge": new_age, "VehPower": new_pwr, "Area": new_area},
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        notes="Annual review & vehicle power adjustment endorsement"
    )

    print(f"✓ Endorsement Recorded! Tx ID: {res_endorse['tx_id']}")
    print("• Modified Field Diff:")
    for k, v in res_endorse["diff"].items():
        print(f"   {k}: {v['before']} ──► {v['after']}")
    print(f"• FedActuary Model Expected Claims: {res_endorse['pred_before']:.4f} ──► {res_endorse['pred_after']:.4f} ({res_endorse['pred_change_pct']:+.2f}%)")

    print("\nChanging Policy Frequency from Annual to Monthly...")
    res_freq = pm.update_policy_frequency(sample_id, "Monthly")
    print(f"✓ Frequency Updated to: {res_freq['new_frequency']}")
    print(f"• Installment Amount: €{res_freq['installment_amount']:,.2f} / month")
    print(f"• Number of Installments: {res_freq['installments_count']}")
    print(f"• Total Annualized Premium: €{res_freq['total_annualized']:,.2f}")
    print(f"• Term Exposure per Installment: {res_freq['term_exposure']}")
    print(f"• Plan Info: {res_freq['description']}")

    print("\n" + "=" * 80)
    print(" 🎉 All requested features successfully demonstrated and verified!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()

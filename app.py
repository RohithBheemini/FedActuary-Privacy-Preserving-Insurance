"""
FedSure Mutual — Enterprise Federated Insurance Platform
A professional, role-based insurance and actuarial management platform
powered by privacy-preserving federated machine learning.
"""

import copy
import datetime
import json
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.claims import ClaimsService
from src.database import (
    add_endorsement, get_all_claims, get_all_endorsements,
    get_all_policies_from_db, get_all_users, get_fl_consortium_rounds,
    get_policy_from_db, get_top_up_history, init_db,
    update_claim_adjudication, update_endorsement_status, upsert_policy_in_db,
    get_endorsements_for_policy
)
from src.data import (
    AREA_MAP, REGIONS, SCALE_COLS, VEH_BRANDS,
    load_preprocessing_artifacts, preprocess_single_record
)
from src.fl_consortium import FLConsortiumService
from src.metrics import gini_coefficient, percentage_deviance_explained
from src.model import MultipleRegression
from src.policy import BILLING_FREQUENCIES, PolicyManager
from src.train import predict

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(
    page_title="FedSure Mutual — Enterprise Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# Professional Corporate Styling (Minimalist, Crisp, No Visual Clutter)
st.markdown("""
<style>
    /* Base Typography & Background */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .main {
        background-color: #F8FAFC;
    }
    
    /* Header Bar */
    .top-header {
        background-color: #0F172A;
        border-bottom: 2px solid #1E293B;
        padding: 16px 24px;
        border-radius: 6px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .top-header-title {
        color: #FFFFFF;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.3px;
        margin: 0;
    }
    .top-header-sub {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-top: 2px;
    }
    .top-header-badge {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #334155;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* Professional Card */
    .pro-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .pro-card-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 8px;
    }
    .pro-card-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0;
    }
    .pro-card-caption {
        font-size: 0.82rem;
        color: #64748B;
    }

    /* Status Badges */
    .status-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-green { background-color: #F0FDF4; color: #166534; border: 1px solid #BBF7D0; }
    .status-blue { background-color: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }
    .status-amber { background-color: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
    .status-red { background-color: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
    .status-slate { background-color: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; }

    /* Policy Wallet Box */
    .policy-spec-box {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-left: 4px solid #2563EB;
        padding: 16px 20px;
        border-radius: 4px;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Caching & Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler, feature_names = load_preprocessing_artifacts(CKPT_DIR)
    return scaler, feature_names


@st.cache_resource
def load_production_model():
    scaler, feature_names = load_artifacts()
    model = MultipleRegression(num_features=len(feature_names))
    model_path = os.path.join(CKPT_DIR, "federated.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def get_policy_manager():
    return PolicyManager()


scaler, feature_names = load_artifacts()
model = load_production_model()
pm = get_policy_manager()


# -----------------------------------------------------------------------------
# Navigation & Routing Architecture
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="padding-bottom: 12px; border-bottom: 1px solid #E2E8F0; margin-bottom: 16px;">
    <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; letter-spacing: -0.3px;">FEDACTUARY ENTERPRISE</div>
    <div style="font-size: 0.78rem; color: #64748B;">FedSure Mutual Group</div>
</div>
""", unsafe_allow_html=True)

# 1. Primary Portal Selector
ROLE_OPTIONS = [
    "Policyholder Portal",
    "Underwriting Workbench",
    "Broker Quoting Portal",
    "Consortium Operations"
]

selected_role = st.sidebar.selectbox(
    "Active Workspace",
    ROLE_OPTIONS,
    index=0
)

st.sidebar.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# 2. Section Navigation conditioned on Role
if selected_role == "Policyholder Portal":
    page_options = [
        "Policy Overview",
        "File a Claim (FNOL)",
        "Policy Endorsement",
        "Coverage Top-Up",
        "Billing Schedule"
    ]
elif selected_role == "Underwriting Workbench":
    page_options = [
        "Endorsement Review Queue",
        "Claims FNOL Adjudication",
        "Risk Rating Engine",
        "Portfolio Analytics",
        "Master Policy Directory"
    ]
elif selected_role == "Broker Quoting Portal":
    page_options = [
        "Quote & Issue Policy",
        "Issued Policies Directory"
    ]
else:  # Consortium Operations
    page_options = [
        "Branch Silos Network",
        "Collaborative Training Round",
        "Model Checkpoint Governance",
        "Actuarial Benchmark Auditing"
    ]

active_page = st.sidebar.radio("Navigation", page_options)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.78rem; color: #64748B; line-height: 1.4;">
    <strong>System State</strong><br>
    Network: Operational<br>
    FL Engine: Active (FedAvg / DP)<br>
    Database: SQLite Enterprise
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Corporate Header Banner
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="top-header">
    <div>
        <div class="top-header-title">{selected_role}</div>
        <div class="top-header-sub">{active_page} &nbsp;|&nbsp; FedSure Mutual Actuarial Management</div>
    </div>
    <div>
        <span class="top-header-badge">CONSORTIUM PRODUCTION ENVIRONMENT</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# WORKSPACE 1: POLICYHOLDER PORTAL
# =============================================================================
if selected_role == "Policyholder Portal":

    # Account Selector
    col_a, col_b = st.columns([2, 1])
    with col_a:
        account_sel = st.selectbox(
            "Select Policyholder Profile",
            [
                "Alice Dupont (Policy #1010996) — Renault Clio B12",
                "Marc Leroy (Policy #1552) — Peugeot 308 B3",
                "Claire Moreau (Policy #424) — Young Driver B1",
                "Lookup Custom Policy ID"
            ]
        )

    if "1010996" in account_sel:
        current_id = 1010996
    elif "1552" in account_sel:
        current_id = 1552
    elif "424" in account_sel:
        current_id = 424
    else:
        with col_b:
            current_id = st.number_input("Enter Policy ID", min_value=1, max_value=9999999, value=1010996, step=1)

    policy_rec = pm.get_policy(current_id)
    if not policy_rec:
        st.error(f"Policy record ID {current_id} not found in repository.")
        st.stop()

    claim_state = pm.get_claim_remaining(current_id)

    # -------------------------------------------------------------------------
    # 1.1 Policy Overview
    # -------------------------------------------------------------------------
    if active_page == "Policy Overview":
        st.markdown("### Contract Specifications & Coverage Status")

        # Specifications Header Box
        st.markdown(f"""
        <div class="policy-spec-box">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <span style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Contract Reference</span>
                    <h3 style="margin: 2px 0 6px 0; color: #0F172A;">POL-{current_id} — {policy_rec.get('holder_name', 'Verified Insured')}</h3>
                    <div style="font-size: 0.85rem; color: #475569;">
                        Vehicle: <strong>{policy_rec.get('VehBrand')}</strong>, {policy_rec.get('VehPower')} CV, Age {policy_rec.get('VehAge')} yrs ({policy_rec.get('VehGas')}) &nbsp;|&nbsp;
                        Driver Age: <strong>{policy_rec.get('DrivAge')}</strong> &nbsp;|&nbsp; Bonus-Malus: <strong>{policy_rec.get('BonusMalus')}</strong>
                    </div>
                </div>
                <div style="text-align: right;">
                    <span class="status-pill status-green">{policy_rec.get('PolicyStatus', 'Active')}</span>
                    <div style="font-size: 0.85rem; color: #475569; margin-top: 6px;">
                        Billing: <strong>{policy_rec.get('PolicyFrequency', 'Annual')}</strong> (€{policy_rec.get('AnnualPremium', 250.0):,.2f}/yr)
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Financial Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Coverage Limit", f"€{claim_state['total_coverage_limit']:,.2f}")
        with c2:
            st.metric("Incurred Claims Payout", f"€{claim_state['total_claim_incurred']:,.2f}")
        with c3:
            st.metric("Remaining Claim Buffer", f"€{claim_state['remaining_claim_amount']:,.2f}")
        with c4:
            st.metric("Pool Utilization", f"{claim_state['claim_utilization_pct']:.1f}%")

        st.progress(min(1.0, claim_state['claim_utilization_pct'] / 100.0))

        st.markdown("#### Historical Records")
        t_clm, t_end, t_top = st.tabs(["Incurred Claims", "Endorsement Audit Log", "Top-Up Deposits"])

        with t_clm:
            claims_data = ClaimsService.get_policy_claims(current_id)
            if claims_data:
                df_c = pd.DataFrame(claims_data)[["claim_id", "incident_date", "claim_type", "amount_claimed", "approved_payout", "status", "triage_recommendation"]]
                df_c.columns = ["Claim ID", "Date", "Incident Type", "Claimed (€)", "Approved (€)", "Status", "Triage Assessment"]
                st.dataframe(df_c, use_container_width=True)
            else:
                st.info("No claims incurred on this policy.")

        with t_end:
            ends = get_endorsements_for_policy(current_id)
            if ends:
                df_e = pd.DataFrame(ends)[["endorsement_id", "requested_at", "endorsement_type", "pred_before", "pred_after", "status"]]
                df_e.columns = ["Endorsement ID", "Timestamp", "Type", "Risk Before", "Risk After", "Status"]
                st.dataframe(df_e, use_container_width=True)
            else:
                st.info("No endorsement records found. Policy operating under original bound parameters.")

        with t_top:
            tops = get_top_up_history(current_id)
            if tops:
                df_t = pd.DataFrame(tops)[["tx_id", "timestamp", "amount", "premium_charge", "new_total_limit", "status"]]
                df_t.columns = ["Transaction ID", "Timestamp", "Amount (€)", "Premium Charge (€)", "New Total Limit (€)", "Status"]
                st.dataframe(df_t, use_container_width=True)
            else:
                st.info("No top-up transactions recorded.")

    # -------------------------------------------------------------------------
    # 1.2 File a Claim (FNOL)
    # -------------------------------------------------------------------------
    elif active_page == "File a Claim (FNOL)":
        st.markdown("### Digital First Notice of Loss (FNOL)")
        st.markdown("Submit claim incident documentation. Claims are evaluated by the Federated AI automated triage protocol.")

        with st.form("clean_fnol_form"):
            col1, col2 = st.columns(2)
            with col1:
                claimant = st.text_input("Claimant Name", value=policy_rec.get("holder_name", "Insured Driver"))
                inc_date = st.date_input("Incident Date", value=datetime.date.today())
                claim_category = st.selectbox("Claim Category", [
                    "Third-Party Collision (Intersection)",
                    "Rear-End Impact",
                    "Windshield Glass & Hail Damage",
                    "Parking Incident",
                    "Theft or Break-In",
                    "Pedestrian / Cyclist Liability"
                ])
            with col2:
                claim_amount = st.number_input("Estimated Claim Amount (€)", min_value=50.0, max_value=150000.0, value=850.0, step=50.0)
                location_text = st.text_input("Incident Location (City / Road)", value="Paris, Boulevard Saint-Germain")
                constat_ref = st.text_input("Police Report or Constat Amiable Reference", value="CONST-2026-092")

            incident_narrative = st.text_area("Incident Statement", value="Low speed impact while navigating roundabout. Damage limited to front bumper and headlight.")
            submit_btn = st.form_submit_button("Submit Claim Report", type="primary")

        if submit_btn:
            with st.spinner("Processing automated triage verification..."):
                res = ClaimsService.file_fnol(
                    id_pol=current_id,
                    claimant_name=claimant,
                    incident_date=str(inc_date),
                    claim_type=claim_category,
                    amount_claimed=claim_amount,
                    description=incident_narrative,
                    policy_details=policy_rec,
                    evidence_attachment=constat_ref
                )
            if res["success"]:
                st.success(f"Claim successfully filed under reference: {res['claim_id']}")
                st.markdown(f"""
                <div class="pro-card">
                    <div class="pro-card-title">Automated Triage Outcome</div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #0F172A;">{res['triage_recommendation']}</div>
                    <div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">
                        Anomaly & Fraud Risk Rating: <strong>{res['fl_fraud_risk_score']*100:.1f}%</strong> | Amount Claimed: <strong>€{claim_amount:,.2f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(res["message"])

        st.markdown("#### Previously Filed Claims")
        prior_claims = ClaimsService.get_policy_claims(current_id)
        if prior_claims:
            df_pc = pd.DataFrame(prior_claims)[["claim_id", "incident_date", "claim_type", "amount_claimed", "approved_payout", "status", "triage_recommendation"]]
            df_pc.columns = ["Claim ID", "Date", "Type", "Amount (€)", "Approved (€)", "Status", "Triage"]
            st.dataframe(df_pc, use_container_width=True)

    # -------------------------------------------------------------------------
    # 1.3 Policy Endorsement
    # -------------------------------------------------------------------------
    elif active_page == "Policy Endorsement":
        st.markdown("### Policy Endorsement & Specification Adjustment")
        st.markdown("Update driver, vehicle, or regional attributes. The federated model automatically re-evaluates risk in real-time.")

        with st.form("clean_mod_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Driver Profile**")
                e_age = st.slider("Driver Age", 18, 90, int(policy_rec.get("DrivAge", 35)))
                e_bm = st.slider("Bonus-Malus", 50, 150, int(policy_rec.get("BonusMalus", 50)))
                e_exp = st.slider("Exposure (Years)", 0.1, 1.0, float(policy_rec.get("Exposure", 1.0)), step=0.05)
            with col2:
                st.markdown("**Vehicle Parameters**")
                v_brand_idx = VEH_BRANDS.index(policy_rec.get("VehBrand", "B12")) if policy_rec.get("VehBrand") in VEH_BRANDS else 0
                e_brand = st.selectbox("Vehicle Brand", VEH_BRANDS, index=v_brand_idx)
                e_power = st.slider("Engine Power (CV)", 4, 15, int(policy_rec.get("VehPower", 6)))
                e_vage = st.slider("Vehicle Age (Years)", 0, 20, int(policy_rec.get("VehAge", 4)))
                e_gas = st.selectbox("Fuel Type", ["Regular", "Diesel"], index=0 if policy_rec.get("VehGas") == "Regular" else 1)
            with col3:
                st.markdown("**Geography & Location**")
                v_area_idx = list(AREA_MAP.keys()).index(policy_rec.get("Area", "C")) if policy_rec.get("Area") in AREA_MAP else 2
                e_area = st.selectbox("Area Code", list(AREA_MAP.keys()), index=v_area_idx)
                v_reg_idx = REGIONS.index(policy_rec.get("Region", "R82")) if policy_rec.get("Region") in REGIONS else 0
                e_region = st.selectbox("Region", REGIONS, index=v_reg_idx)
                e_dens = st.number_input("Population Density (hab/km²)", min_value=1.0, max_value=30000.0, value=float(policy_rec.get("Density", 1000.0)), step=100.0)

            e_notes = st.text_input("Endorsement Reason", value="Vehicle upgrade and change of garaging postal code.")
            apply_mod_btn = st.form_submit_button("Preview Risk & Apply Endorsement", type="primary")

        if apply_mod_btn:
            updates = {
                "DrivAge": e_age, "BonusMalus": e_bm, "Exposure": e_exp,
                "VehBrand": e_brand, "VehPower": e_power, "VehAge": e_vage,
                "VehGas": e_gas, "Area": e_area, "Region": e_region, "Density": e_dens
            }
            res = pm.update_policy_details(
                id_pol=current_id,
                updates=updates,
                model=model,
                scaler=scaler,
                feature_names=feature_names,
                notes=e_notes
            )
            if res["success"]:
                st.success(f"Endorsement {res['tx_id']} applied successfully.")
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Expected Claims / Year (Before)", f"{res['pred_before']:.4f}")
                with col_m2:
                    st.metric("Expected Claims / Year (After)", f"{res['pred_after']:.4f}", delta=f"{res['pred_change_pct']:+.2f}%")
                with col_m3:
                    st.metric("Risk Classification", "Low Risk" if res['pred_after'] < 0.08 else ("Standard Risk" if res['pred_after'] < 0.13 else "Elevated Risk"))

                st.json(res["diff"])

    # -------------------------------------------------------------------------
    # 1.4 Coverage Top-Up
    # -------------------------------------------------------------------------
    elif active_page == "Coverage Top-Up":
        st.markdown("### Coverage Limit Top-Up")
        st.markdown("Increase policy coverage buffer on demand. Actuarial pricing: **0.40% baseline** adjusted for driver Bonus-Malus.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="pro-card">
                <div class="pro-card-title">Active Coverage Limit</div>
                <div class="pro-card-value">€{claim_state['total_coverage_limit']:,.2f}</div>
                <div class="pro-card-caption">
                    Remaining claim buffer: <strong>€{claim_state['remaining_claim_amount']:,.2f}</strong> ({100-claim_state['claim_utilization_pct']:.1f}% remaining)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            top_up_choice = st.radio("Select Top-Up Amount", [5000.0, 10000.0, 20000.0, 50000.0], format_func=lambda x: f"+€{x:,.0f} Additional Limit")
            bm_factor = float(policy_rec.get("BonusMalus", 100)) / 100.0
            charge = round(top_up_choice * 0.004 * bm_factor, 2)
            st.markdown(f"""
            <div class="pro-card">
                <div class="pro-card-title">Premium Surcharge</div>
                <div class="pro-card-value" style="color: #2563EB;">€{charge:,.2f}</div>
                <div class="pro-card-caption">One-time payment. Coverage buffer updates immediately upon confirmation.</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Confirm and Apply Top-Up", type="primary"):
            res = pm.apply_top_up(current_id, top_up_choice, notes="Self-Service Portal Top-Up")
            if res["success"]:
                st.success(f"Top-Up completed (Reference: {res['tx_id']}). New total coverage limit: €{res['new_total_limit']:,.2f}.")

    # -------------------------------------------------------------------------
    # 1.5 Billing Schedule
    # -------------------------------------------------------------------------
    elif active_page == "Billing Schedule":
        st.markdown("### Payment Schedule & Frequency")
        curr_freq = policy_rec.get("PolicyFrequency", "Annual")
        st.write(f"Active Schedule: **{curr_freq}**")

        new_freq = st.selectbox("Select Billing Frequency", list(BILLING_FREQUENCIES.keys()), index=list(BILLING_FREQUENCIES.keys()).index(curr_freq))
        freq_info = BILLING_FREQUENCIES[new_freq]

        ann_prem = float(policy_rec.get("AnnualPremium", 250.0))
        inst_amt = round(ann_prem * freq_info["installment_factor"], 2)
        tot_ann = round(inst_amt * freq_info["installments_per_year"], 2)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Installments per Year", f"{freq_info['installments_per_year']}")
        with c2:
            st.metric("Per Installment Amount", f"€{inst_amt:,.2f}")
        with c3:
            st.metric("Total Annualized Premium", f"€{tot_ann:,.2f}")

        if st.button("Update Billing Schedule", type="primary"):
            res = pm.update_policy_frequency(current_id, new_freq)
            if res["success"]:
                st.success(f"Billing frequency updated to {new_freq} (Transaction: {res['tx_id']}).")


# =============================================================================
# WORKSPACE 2: UNDERWRITING WORKBENCH
# =============================================================================
elif selected_role == "Underwriting Workbench":

    if active_page == "Endorsement Review Queue":
        st.markdown("### Pending Endorsement Requests")
        all_ends = get_all_endorsements()

        if all_ends:
            df_e = pd.DataFrame(all_ends)[["endorsement_id", "id_pol", "requested_by", "requested_at", "endorsement_type", "pred_before", "pred_after", "status"]]
            df_e.columns = ["ID", "Policy", "Requested By", "Timestamp", "Type", "Risk Before", "Risk After", "Status"]
            st.dataframe(df_e, use_container_width=True)

            st.markdown("#### Review Selected Request")
            selected_eid = st.selectbox("Select Endorsement Reference", [e["endorsement_id"] for e in all_ends])
            target_e = next(e for e in all_ends if e["endorsement_id"] == selected_eid)

            st.write(f"Policy: **POL-{target_e['id_pol']}** | Type: **{target_e['endorsement_type']}** | Status: **{target_e['status']}**")
            st.code(target_e["changes_json"], language="json")

            col1, col2 = st.columns(2)
            with col1:
                decision = st.selectbox("Adjudication Action", ["Approved", "Rejected", "Pending Review"])
            with col2:
                notes = st.text_input("Underwriter Notes", value="Verified against national licensing database. Approved.")

            if st.button("Finalize Decision", type="primary"):
                update_endorsement_status(selected_eid, decision, reviewed_by="Sarah Jenkins (Lead Underwriter)", notes=notes)
                st.success(f"Endorsement {selected_eid} marked as {decision}.")
        else:
            st.info("No endorsement requests pending review.")

    elif active_page == "Claims FNOL Adjudication":
        st.markdown("### Claims Adjudication Desk")
        all_clms = ClaimsService.list_all_claims()

        if all_clms:
            df_c = pd.DataFrame(all_clms)[["claim_id", "id_pol", "claimant_name", "incident_date", "claim_type", "amount_claimed", "approved_payout", "status", "fl_fraud_risk_score", "triage_recommendation"]]
            df_c.columns = ["Claim ID", "Policy", "Claimant", "Date", "Type", "Claimed (€)", "Approved (€)", "Status", "FL Risk", "Triage"]
            st.dataframe(df_c, use_container_width=True)

            st.markdown("#### Review & Adjudicate Claim")
            selected_cid = st.selectbox("Select Claim", [c["claim_id"] for c in all_clms])
            target_c = next(c for c in all_clms if c["claim_id"] == selected_cid)

            st.markdown(f"""
            <div class="pro-card">
                <div class="pro-card-title">Claim {target_c['claim_id']} — Policy #{target_c['id_pol']}</div>
                <div style="font-size: 0.95rem; color: #0F172A;">
                    <strong>Claimant:</strong> {target_c['claimant_name']} | <strong>Date:</strong> {target_c['incident_date']} | <strong>Type:</strong> {target_c['claim_type']}
                </div>
                <div style="margin: 8px 0; color: #334155;"><strong>Statement:</strong> {target_c['description']}</div>
                <div style="font-size: 0.85rem; color: #475569;">
                    AI Triage Rating: <strong>{target_c['fl_fraud_risk_score']*100:.1f}% Anomaly Score</strong> ({target_c['triage_recommendation']})
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                action = st.selectbox("Adjudication Decision", ["Approved", "Settled", "Denied", "Under Review"])
                payout = st.number_input("Approved Payout Amount (€)", min_value=0.0, max_value=float(target_c["amount_claimed"]), value=float(target_c["amount_claimed"]))
            with col2:
                notes = st.text_area("Settlement Audit Notes", value="Settled in accordance with repair invoice.")

            if st.button("Confirm Claim Adjudication", type="primary"):
                res = ClaimsService.adjudicate(selected_cid, action, payout, notes)
                if res["success"]:
                    st.success(f"Claim {selected_cid} updated to {action} with approved payout of €{payout:,.2f}.")
        else:
            st.info("No claims records found.")

    elif active_page == "Risk Rating Engine":
        st.markdown("### Federated Model Risk Rating & SHAP Attribution")
        st.markdown("Real-time actuarial expected claim frequency estimation.")

        col1, col2, col3 = st.columns(3)
        with col1:
            u_age = st.slider("Driver Age", 18, 90, 35)
            u_bm = st.slider("Bonus-Malus Index", 50, 150, 50)
            u_exp = st.slider("Exposure (Years)", 0.1, 1.0, 1.0, step=0.05)
        with col2:
            u_brand = st.selectbox("Vehicle Brand", VEH_BRANDS, index=3)
            u_pwr = st.slider("Engine Power (CV)", 4, 15, 6)
            u_vage = st.slider("Vehicle Age (Years)", 0, 20, 3)
            u_gas = st.selectbox("Fuel Type", ["Regular", "Diesel"], index=0)
        with col3:
            u_area = st.selectbox("Area Code", list(AREA_MAP.keys()), index=2)
            u_reg = st.selectbox("Region Code", REGIONS, index=17)
            u_dens = st.number_input("Density (hab/km²)", 1.0, 30000.0, 1200.0, step=100.0)

        raw_rec = {
            "Exposure": u_exp, "Area": u_area, "VehPower": u_pwr, "VehAge": u_vage,
            "DrivAge": u_age, "BonusMalus": u_bm, "VehBrand": u_brand, "VehGas": u_gas,
            "Density": u_dens, "Region": u_reg
        }
        arr = preprocess_single_record(raw_rec, scaler, feature_names)
        pred_val = float(predict(model, arr)[0])

        if pred_val < 0.07:
            tier, pill_class = "Tier 1: Preferred Clean Risk", "status-green"
        elif pred_val < 0.12:
            tier, pill_class = "Tier 2: Standard Commercial Risk", "status-blue"
        elif pred_val < 0.18:
            tier, pill_class = "Tier 3: Elevated Risk", "status-amber"
        else:
            tier, pill_class = "Tier 4: Substandard Risk", "status-red"

        st.markdown(f"""
        <div class="pro-card" style="margin-top: 16px;">
            <div class="pro-card-title">Model Estimation Output</div>
            <div class="pro-card-value">{pred_val:.4f} <span style="font-size: 1rem; color: #64748B;">expected claims / yr</span></div>
            <div style="margin-top: 6px;"><span class="status-pill {pill_class}">{tier}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Actuarial Component Attribution")
        shap_proxies = {
            "BonusMalus Surcharge": (u_bm - 50) * 0.0012,
            "Engine Power Load": (u_pwr - 6) * 0.004,
            "Driver Inexperience Factor": max(0, 30 - u_age) * 0.003,
            "Urban Density Load": np.log(max(1, u_dens)) * 0.005,
            "Vehicle Age Discount": -u_vage * 0.002
        }
        df_sh = pd.DataFrame(list(shap_proxies.items()), columns=["Factor", "Net Frequency Impact"])
        fig = px.bar(df_sh, x="Net Frequency Impact", y="Factor", orientation="h", color="Net Frequency Impact",
                     color_continuous_scale="Blues", title="Risk Factor Decomposition")
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    elif active_page == "Portfolio Analytics":
        st.markdown("### Enterprise Portfolio Claims & Loss Metrics")
        stats = pm.get_portfolio_stats()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Policies in Portfolio", f"{stats['total_policies']:,}")
        with c2:
            st.metric("Total Claims Generated", f"{stats['total_claims_count']:,}")
        with c3:
            st.metric("Total Incurred Claim Losses", f"€{stats['total_claim_payout']:,.2f}")
        with c4:
            st.metric("Mean Severity per Claim", f"€{stats['average_severity']:,.2f}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            reg_df = pd.DataFrame(list(stats["distribution_by_region"].items()), columns=["Region", "Policies"]).sort_values("Policies", ascending=False).head(10)
            fig_r = px.bar(reg_df, x="Region", y="Policies", title="Top 10 Regions by Policy Count")
            fig_r.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_r, use_container_width=True)
        with col2:
            age_df = pd.DataFrame(list(stats["distribution_by_driver_age_group"].items()), columns=["Age Bracket", "Policies"])
            fig_a = px.pie(age_df, names="Age Bracket", values="Policies", title="Driver Age Demographic Distribution")
            fig_a.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_a, use_container_width=True)

    elif active_page == "Master Policy Directory":
        st.markdown("### Portfolio Dataset Search & Pagination")
        col1, col2, col3 = st.columns(3)
        with col1:
            f_reg = st.selectbox("Region Filter", ["All"] + REGIONS)
        with col2:
            f_stat = st.selectbox("Status Filter", ["All", "Active (Clean)", "Active with Claims", "Critical (Near Limit)", "Limit Exhausted"])
        with col3:
            page = st.number_input("Page", min_value=1, value=1, step=1)

        p_reg = None if f_reg == "All" else f_reg
        p_stat = None if f_stat == "All" else f_stat

        df_p, total_cnt = pm.query_policies(region=p_reg, policy_status=p_stat, page=page, page_size=20)
        st.caption(f"Displaying page {page} of {int(np.ceil(total_cnt / 20))} ({total_cnt:,} total records matching filters)")
        st.dataframe(df_p[["IDpol", "DrivAge", "BonusMalus", "VehBrand", "VehPower", "Region", "ClaimNb", "TotalClaimAmount", "TotalCoverageLimit", "RemainingClaimAmount", "PolicyStatus"]], use_container_width=True)


# =============================================================================
# WORKSPACE 3: BROKER QUOTING PORTAL
# =============================================================================
elif selected_role == "Broker Quoting Portal":

    if active_page == "Quote & Issue Policy":
        st.markdown("### Commercial Quote & Policy Issuance Engine")
        st.markdown("Evaluate driver and vehicle risk, select coverage tier, and issue contract directly to the enterprise database.")

        with st.form("clean_quote_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Client Information**")
                c_name = st.text_input("Full Name", value="Jean-Michel Dupont")
                c_email = st.text_input("Email Address", value="jm.dupont@client.fr")
                c_age = st.slider("Driver Age", 18, 90, 42)
                c_bm = st.slider("Bonus-Malus Rating", 50, 150, 50)
            with col2:
                st.markdown("**Vehicle & Coverage Tier**")
                c_brand = st.selectbox("Vehicle Brand", VEH_BRANDS, index=0)
                c_power = st.slider("Engine Power (CV)", 4, 15, 6)
                c_vage = st.slider("Vehicle Age (Years)", 0, 20, 2)
                c_gas = st.selectbox("Fuel Type", ["Regular", "Diesel"], index=0)
                c_reg = st.selectbox("Region Code", REGIONS, index=11)
                tier_sel = st.selectbox("Coverage Limit Tier", ["Silver (€30,000)", "Gold (€50,000 Standard)", "Platinum (€100,000)"])

            calc_quote = st.form_submit_button("Generate Actuarial Quote", type="primary")

        if calc_quote:
            raw_rec = {
                "Exposure": 1.0, "Area": "C", "VehPower": c_power, "VehAge": c_vage,
                "DrivAge": c_age, "BonusMalus": c_bm, "VehBrand": c_brand, "VehGas": c_gas,
                "Density": 850.0, "Region": c_reg
            }
            arr = preprocess_single_record(raw_rec, scaler, feature_names)
            pred_f = float(predict(model, arr)[0])

            base_rate = 250.0 * (c_bm / 100.0) * (c_power / 6.0)
            if "Silver" in tier_sel:
                lim_amt, mult = 30000.0, 0.85
            elif "Platinum" in tier_sel:
                lim_amt, mult = 100000.0, 1.35
            else:
                lim_amt, mult = 50000.0, 1.0

            ann_quote = round(base_rate * mult * (1.0 + pred_f), 2)
            mon_quote = round(ann_quote / 12.0 * 1.06, 2)

            st.markdown(f"""
            <div class="pro-card" style="margin-top: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="pro-card-title">Quote Generated</div>
                        <h3 style="margin: 2px 0 4px 0; color: #0F172A;">{tier_sel} for {c_name}</h3>
                        <div style="font-size: 0.85rem; color: #64748B;">Predicted Claim Frequency: <strong>{pred_f:.4f} claims/yr</strong></div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.8rem; font-weight: 700; color: #0F172A;">€{ann_quote:,.2f} <span style="font-size: 0.85rem; color: #64748B;">/ yr</span></div>
                        <div style="font-size: 0.9rem; color: #166534; font-weight: 600;">or €{mon_quote:,.2f} / month</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            bind_id = int(time.time()) % 1000000 + 400000
            if st.button(f"Bind Contract and Issue Policy POL-{bind_id}", type="primary"):
                p_dict = {
                    "id_pol": bind_id,
                    "holder_name": c_name,
                    "holder_email": c_email,
                    "driv_age": c_age,
                    "bonus_malus": c_bm,
                    "exposure": 1.0,
                    "veh_power": c_power,
                    "veh_age": c_vage,
                    "veh_gas": c_gas,
                    "veh_brand": c_brand,
                    "area": "C",
                    "region": c_reg,
                    "density": 850.0,
                    "base_limit": lim_amt,
                    "top_up_amount": 0.0,
                    "total_limit": lim_amt,
                    "frequency": "Annual",
                    "annual_premium": ann_quote,
                    "status": "Active (Clean)",
                    "fl_predicted_claim_freq": pred_f,
                    "fl_risk_tier": "Standard Risk"
                }
                upsert_policy_in_db(p_dict)
                st.success(f"Policy POL-{bind_id} bound and active for {c_name}!")

    elif active_page == "Issued Policies Directory":
        st.markdown("### Broker Issued Policies Directory")
        db_policies = get_all_policies_from_db()
        if db_policies:
            df_bp = pd.DataFrame(db_policies)[["id_pol", "holder_name", "holder_email", "driv_age", "bonus_malus", "veh_brand", "total_limit", "annual_premium", "frequency", "status", "updated_at"]]
            df_bp.columns = ["Policy ID", "Holder Name", "Email", "Age", "BM", "Brand", "Coverage Limit (€)", "Premium (€)", "Billing", "Status", "Last Modified"]
            st.dataframe(df_bp, use_container_width=True)
        else:
            st.info("No policies recorded in database.")


# =============================================================================
# WORKSPACE 4: CONSORTIUM OPERATIONS
# =============================================================================
elif selected_role == "Consortium Operations":

    if active_page == "Branch Silos Network":
        st.markdown("### Decentralized Regional Silo Topology")
        st.markdown("Collaborative federated network topology across isolated regional carrier repositories.")
        overview = FLConsortiumService.get_consortium_overview()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Connected Regional Silos", f"{len(overview['branches'])} Branches")
        with c2:
            st.metric("Total Decentralized Records", f"{overview['total_isolated_samples']:,}", delta="Zero Raw Data Pooling")
        with c3:
            st.metric("Active Production Round", f"Round #{overview['current_production_round']}")

        st.markdown("#### Participating Regional Nodes")
        for b in overview["branches"]:
            st.markdown(f"""
            <div class="pro-card" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A;">{b['name']}</div>
                    <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">
                        Region Code: <strong>{b['region']}</strong> | Local Worker Nodes: <strong>{b['nodes']}</strong>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A;">{b['data_rows']:,} rows</div>
                    <span class="status-pill status-green">{b['status']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif active_page == "Collaborative Training Round":
        st.markdown("### Orchestrate Federated Training Iteration")
        st.markdown("Trigger a synchronized FedAvg / FedProx round with Differential Privacy gradient perturbation.")

        col1, col2, col3 = st.columns(3)
        with col1:
            strategy = st.selectbox("Aggregation Strategy", ["FedAvg", "FedProx (Proximal Regularization)"])
        with col2:
            dp_setting = st.select_slider("DP Privacy Budget (ε)", options=[1.0, 3.0, 8.0, 99.0], value=3.0, format_func=lambda x: "ε = ∞ (No DP)" if x > 10 else f"ε = {x:.1f}")
        with col3:
            epochs = st.slider("Local Epochs per Silo", 1, 5, 2)

        eps_target = None if dp_setting > 10 else dp_setting

        if st.button("Execute Collaborative Federated Round", type="primary"):
            p_bar = st.progress(0)
            status_placeholder = st.empty()

            def progress_hook(pct, message):
                p_bar.progress(pct)
                status_placeholder.text(message)

            res = FLConsortiumService.trigger_consortium_round(
                strategy=strategy,
                epochs=epochs,
                dp_epsilon=eps_target,
                progress_callback=progress_hook
            )
            time.sleep(0.4)
            st.success(res["message"])

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("New Global Round", f"Round #{res['round_number']}")
            with c2:
                st.metric("Training Loss", f"{res['train_loss']:.4f}")
            with c3:
                st.metric("Gini Rank Coefficient", f"{res['gini_score']:.4f}")

    elif active_page == "Model Checkpoint Governance":
        st.markdown("### Checkpoint Audit Logs & Production Promotion")
        rounds_data = FLConsortiumService.get_round_history()

        if rounds_data:
            df_r = pd.DataFrame(rounds_data)[["round_id", "round_number", "timestamp", "aggregation_strategy", "dp_epsilon", "train_loss", "gini_score", "is_production"]]
            df_r.columns = ["Round ID", "Round #", "Timestamp", "Strategy", "DP ε", "Loss", "Gini", "Production"]
            st.dataframe(df_r, use_container_width=True)

            st.markdown("#### Hot-Swap Production Model Checkpoint")
            sel_round = st.selectbox("Select Checkpoint to Promote", [r["round_id"] for r in rounds_data])

            if st.button("Promote Checkpoint to Active Production", type="primary"):
                success = FLConsortiumService.promote_checkpoint(sel_round)
                if success:
                    st.success(f"Checkpoint {sel_round} promoted to active production. Rating engines will load updated weights on next inference.")
                else:
                    st.error("Promotion failed.")

    elif active_page == "Actuarial Benchmark Auditing":
        st.markdown("### Peer-Reviewed Benchmark Verification")
        st.markdown("Reproduction benchmarks adhering to **Śmietanka et al. (British Actuarial Journal, 2026)** across Non-IID Dirichlet distribution, Differential Privacy trade-offs, and SHAP explainability.")

        t1, t2, t3 = st.tabs(["Data Heterogeneity (Non-IID)", "Differential Privacy (Opacus DP)", "Feature Attribution Consistency"])

        with t1:
            h_path = os.path.join(RESULTS_DIR, "phase2_heterogeneity.json")
            if os.path.exists(h_path):
                with open(h_path, "r") as f:
                    h_data = json.load(f)
                runs = h_data.get("label_shift_claimnb", [])
                if runs:
                    df_h = pd.DataFrame([{"Dirichlet Alpha": r["alpha"], "Algorithm": r["method"], "%PDE": r["final_pde"], "Gini": r["final_gini"]} for r in runs])
                    st.dataframe(df_h, use_container_width=True)

        with t2:
            p_path = os.path.join(RESULTS_DIR, "phase3_privacy_utility.json")
            if os.path.exists(p_path):
                with open(p_path, "r") as f:
                    p_data = json.load(f)
                dp_runs = p_data.get("flat_clipping", [])
                df_dp = pd.DataFrame([{"Target ε": r["target_epsilon"], "Realized ε": round(r["realized_epsilon"], 2), "%PDE": round(r["pde"], 3), "Gini": round(r["gini"], 3)} for r in dp_runs])
                st.dataframe(df_dp, use_container_width=True)

        with t3:
            r_path = os.path.join(RESULTS_DIR, "feature_rank.json")
            if os.path.exists(r_path):
                with open(r_path, "r") as f:
                    r_data = json.load(f)
                top_feats = r_data.get("feature_rankings", {}).get("global", [])[:10]
                df_rf = pd.DataFrame(top_feats, columns=["Actuarial Feature", "Mean |SHAP| Value"])
                fig_rf = px.bar(df_rf, x="Mean |SHAP| Value", y="Actuarial Feature", orientation="h", title="Top 10 Global Actuarial Risk Predictors")
                fig_rf.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_rf, use_container_width=True)

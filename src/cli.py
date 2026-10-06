"""
FedActuary Enterprise CLI: Unified command-line interface for
platform management, actuarial predictions, testing, and federated learning operations.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import (
    CKPT_DIR, DATA_DIR, DEFAULT_SERVER_PORT, FEDERATED_CKPT,
    FREQ_DATA_PATH, PROJECT_ROOT, SEV_DATA_PATH, DB_PATH
)


def cmd_info(args):
    """Displays platform status, dataset statistics, models, and consortium state."""
    print("=" * 70)
    print("  FEDACTUARY ENTERPRISE PLATFORM — SYSTEM STATUS")
    print("=" * 70)

    # 1. Paths & Environment
    print(f"\n[Environment]")
    print(f"  • Project Directory: {PROJECT_ROOT}")
    print(f"  • Python Runtime:    {sys.version.split()[0]} ({sys.executable})")

    # 2. Datasets
    print(f"\n[Datasets]")
    freq_exists = os.path.exists(FREQ_DATA_PATH)
    sev_exists = os.path.exists(SEV_DATA_PATH)
    print(f"  • freMTPL2freq.csv: {'READY' if freq_exists else 'MISSING'} ({FREQ_DATA_PATH})")
    print(f"  • freMTPL2sev.csv:  {'READY' if sev_exists else 'MISSING'} ({SEV_DATA_PATH})")

    if freq_exists and sev_exists:
        try:
            from src.data import load_severity_data
            df_sev = load_severity_data()
            print(f"    - Severity claims recorded: {len(df_sev):,} claims (€{df_sev['ClaimAmount'].sum():,.2f})")
        except Exception as e:
            print(f"    - Note: Error inspecting severity: {e}")

    # 3. Checkpoints
    print(f"\n[Model Checkpoints]")
    ckpts = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")] if os.path.exists(CKPT_DIR) else []
    print(f"  • Production model: {'READY' if os.path.exists(FEDERATED_CKPT) else 'NOT FOUND'} ({FEDERATED_CKPT})")
    print(f"  • Total checkpoints available: {len(ckpts)}")
    for c in ckpts[:5]:
        print(f"    - {c}")
    if len(ckpts) > 5:
        print(f"    ... and {len(ckpts) - 5} more")

    # 4. Database
    print(f"\n[Enterprise Database]")
    print(f"  • SQLite Database:  {'READY' if os.path.exists(DB_PATH) else 'NOT FOUND'} ({DB_PATH})")
    if os.path.exists(DB_PATH):
        try:
            from src.database import get_all_users, get_all_policies_from_db, get_all_claims
            users = get_all_users()
            policies = get_all_policies_from_db()
            claims = get_all_claims()
            print(f"    - Users:    {len(users)}")
            print(f"    - Policies: {len(policies)}")
            print(f"    - Claims:   {len(claims)}")
        except Exception as e:
            print(f"    - Note: Error reading DB: {e}")

    # 5. Consortium Overview
    print(f"\n[Federated Consortium]")
    try:
        from src.fl_consortium import FLConsortiumService
        consortium = FLConsortiumService.get_consortium_overview()
        print(f"  • Name:             {consortium['consortium_name']}")
        print(f"  • Regional Silos:   {len(consortium['branches'])} branches ({consortium['total_isolated_samples']:,} isolated rows)")
        print(f"  • Active Protocol:  {consortium['consensus_protocol']}")
        print(f"  • Production Model: {consortium['active_model_checkpoint']}")
    except Exception as e:
        print(f"  • Status: Unable to query consortium ({e})")

    print("\n" + "=" * 70)


def cmd_serve(args):
    """Launches the Streamlit interactive dashboard."""
    port = args.port or DEFAULT_SERVER_PORT
    print(f"Starting FedSure Mutual Platform on http://localhost:{port} ...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port)]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def cmd_verify(args):
    """Executes the full 10-step production workflow verification."""
    from verify_workflow import test_production_workflow
    test_production_workflow()


def cmd_test(args):
    """Runs the full unit test suite."""
    print("Running FedActuary Unit Test Suite...")
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(res.returncode)


def cmd_predict(args):
    """Evaluates expected claim frequency for given policy parameters."""
    from src.data import load_preprocessing_artifacts, preprocess_single_record
    from src.model import MultipleRegression
    import torch

    scaler, feature_names = load_preprocessing_artifacts(CKPT_DIR)
    model = MultipleRegression(num_features=len(feature_names))
    model.load_state_dict(torch.load(FEDERATED_CKPT, map_location="cpu"))
    model.eval()

    sample = {
        "Exposure": args.exposure,
        "Area": args.area,
        "VehPower": args.veh_power,
        "VehAge": args.veh_age,
        "DrivAge": args.driv_age,
        "BonusMalus": args.bonus_malus,
        "VehBrand": args.veh_brand,
        "VehGas": args.veh_gas,
        "Density": args.density,
        "Region": args.region
    }

    x_arr = preprocess_single_record(sample, scaler, feature_names)
    with torch.no_grad():
        x_t = torch.tensor(x_arr, dtype=torch.float32)
        pred_claims = float(model(x_t).item())
        annual_freq = pred_claims / max(args.exposure, 0.001)

    print("\n" + "=" * 60)
    print("  FEDACTUARY NEURAL NETWORK PREDICTION")
    print("=" * 60)
    print(f"  • Driver Age:           {args.driv_age} yrs")
    print(f"  • Bonus-Malus:          {args.bonus_malus}")
    print(f"  • Vehicle Power / Age:  {args.veh_power} cv / {args.veh_age} yrs")
    print(f"  • Geographic:           Area {args.area}, Region {args.region}")
    print(f"  • Exposure Term:        {args.exposure} yr(s)")
    print("-" * 60)
    print(f"  ★ Expected Claim Count: {pred_claims:.4f} claims")
    print(f"  ★ Annual Claim Rate:    {annual_freq*100:.2f}% per year")
    print("=" * 60 + "\n")


def cmd_policy(args):
    """Queries policy details, claims incurred, and remaining limit."""
    from src.policy import PolicyManager
    pm = PolicyManager()
    info = pm.get_claim_remaining(args.id_pol)

    print("\n" + "=" * 65)
    print(f"  POLICY COVERAGE & CLAIMS AUDIT: ID #{args.id_pol}")
    print("=" * 65)
    print(f"  • Base Coverage Limit:       €{info['base_coverage_limit']:,.2f}")
    print(f"  • Top-Up Amount Added:       €{info['top_up_amount']:,.2f}")
    print(f"  • Total Limit:               €{info['total_coverage_limit']:,.2f}")
    print(f"  • Claims Filed:              {info['total_claims_generated']} event(s)")
    print(f"  • Incurred Claims Losses:    €{info['total_claim_incurred']:,.2f}")
    print(f"  • REMAINING CLAIM BALANCE:   €{info['remaining_claim_amount']:,.2f} ({info['remaining_claim_pct']:.1f}% remaining)")
    print(f"  • Utilization Rate:          {info['claim_utilization_pct']:.1f}%")
    print(f"  • Status:                    {info['status_description']}")
    print("=" * 65 + "\n")


def cmd_top_up(args):
    """Applies a coverage top-up to a policy."""
    from src.policy import PolicyManager
    pm = PolicyManager()
    res = pm.apply_top_up(args.id_pol, args.amount, notes=args.notes or "CLI top-up")

    print("\n" + "=" * 65)
    if res["success"]:
        print(f"  ✓ TOP-UP SUCCESSFULLY APPLIED!")
        print(f"  • Transaction ID:     {res['tx_id']}")
        print(f"  • Policy ID:          #{args.id_pol}")
        prem_val = res.get('premium_charge', res.get('prorated_premium', 0.0))
        print(f"  • Premium Charged:    €{prem_val:,.2f}")
        print(f"  • New Total Limit:    €{res['new_total_limit']:,.2f}")
        print(f"  • New Remaining:      €{res['new_remaining_claim']:,.2f}")
    else:
        print(f"  ✗ TOP-UP FAILED: {res.get('message', 'Unknown error')}")
    print("=" * 65 + "\n")


def cmd_fl_round(args):
    """Triggers a live multi-branch federated learning round."""
    from src.fl_consortium import FLConsortiumService
    print(f"Initiating Federated Consortium Round ({args.strategy}, DP ε={args.epsilon})...")

    def progress(pct, msg):
        print(f"  [{pct}%] {msg}")

    res = FLConsortiumService.trigger_consortium_round(
        strategy=args.strategy,
        epochs=args.epochs,
        dp_epsilon=args.epsilon,
        progress_callback=progress
    )
    print("\n" + "=" * 65)
    print(f"  ✓ FEDERATED LEARNING ROUND #{res['round_number']} COMPLETED")
    print(f"  • Participating Silos:  {res['branches_count']} regional branches")
    print(f"  • Final Training Loss:  {res['train_loss']:.4f}")
    print(f"  • Gini Ranking Score:   {res['gini_score']:.4f}")
    print(f"  • Checkpoint Promoted:  {res['checkpoint_path']}")
    print("=" * 65 + "\n")


def cmd_export_guide(args):
    """Generates the enterprise platform guide PDF."""
    cmd = [sys.executable, "generate_pdf_guide.py"]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode == 0:
        print("✓ Enterprise Platform Guide PDF generated successfully.")


def main():
    parser = argparse.ArgumentParser(
        prog="fedactuary",
        description="FedActuary Enterprise CLI — Privacy-Preserving Federated Insurance Management"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    subparsers.add_parser("info", help="Display system status, datasets, models, and consortium state")

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch Streamlit interactive web platform")
    p_serve.add_argument("--port", type=int, default=8501, help="Port to serve on (default: 8501)")

    # verify
    subparsers.add_parser("verify", help="Execute full 10-step production workflow verification")

    # test
    subparsers.add_parser("test", help="Execute unit test suite")

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict expected claims frequency for policy inputs")
    p_pred.add_argument("--driv-age", type=int, default=35, help="Driver Age (years)")
    p_pred.add_argument("--bonus-malus", type=int, default=50, help="Bonus-Malus score (50-150)")
    p_pred.add_argument("--veh-power", type=int, default=6, help="Vehicle power (CV)")
    p_pred.add_argument("--veh-age", type=int, default=4, help="Vehicle age (years)")
    p_pred.add_argument("--area", type=str, default="C", help="Area category (A-F)")
    p_pred.add_argument("--region", type=str, default="R82", help="Region code (e.g. R82, R11)")
    p_pred.add_argument("--density", type=float, default=1000.0, help="Population density")
    p_pred.add_argument("--veh-gas", type=str, default="Regular", choices=["Regular", "Diesel"], help="Fuel type")
    p_pred.add_argument("--veh-brand", type=str, default="B12", help="Vehicle brand (e.g. B12, B1)")
    p_pred.add_argument("--exposure", type=float, default=1.0, help="Policy exposure period (fraction of year)")

    # policy
    p_pol = subparsers.add_parser("policy", help="Inspect policy limits, claims, and remaining coverage")
    p_pol.add_argument("--id-pol", type=int, required=True, help="Policy ID (e.g. 424, 1552)")

    # top-up
    p_top = subparsers.add_parser("top-up", help="Apply coverage top-up to a policy")
    p_top.add_argument("--id-pol", type=int, required=True, help="Policy ID")
    p_top.add_argument("--amount", type=float, required=True, help="Top-up limit to add in EUR")
    p_top.add_argument("--notes", type=str, default="", help="Optional notes")

    # fl-round
    p_fl = subparsers.add_parser("fl-round", help="Trigger a multi-branch federated learning round")
    p_fl.add_argument("--strategy", type=str, default="FedAvg", choices=["FedAvg", "FedProx"], help="Aggregation strategy")
    p_fl.add_argument("--epochs", type=int, default=2, help="Local epochs per branch")
    p_fl.add_argument("--epsilon", type=float, default=3.0, help="Differential privacy budget epsilon")

    # export-guide
    subparsers.add_parser("export-guide", help="Generate FedSure Mutual Enterprise Platform Guide PDF")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "info": cmd_info,
        "serve": cmd_serve,
        "verify": cmd_verify,
        "test": cmd_test,
        "predict": cmd_predict,
        "policy": cmd_policy,
        "top-up": cmd_top_up,
        "fl-round": cmd_fl_round,
        "export-guide": cmd_export_guide,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

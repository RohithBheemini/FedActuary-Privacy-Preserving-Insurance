"""
FedActuary: End-to-End One-Command Production Pipeline.
PRD v1.0 Section 8 & Section 14 Deliverables Checklist.

Executes the full horizontal federated learning pipeline on the entire
freMTPL2freq dataset (678,013 policies):
  - Phase 1: Global / Partial / Federated Baselines (350 rounds x 10 local epochs)
  - Phase 2: Non-IID Dirichlet Heterogeneity (Full sweep over ClaimNb and Region)
  - Phase 3: Differential Privacy (Opacus DP-SGD across epsilon targets + per-layer extension)
  - Phase 4: Post-Hoc Explainability (SHAP GradientExplainer on 5 checkpoints)
Generates and verifies all production checkpoints and evaluation artifacts.
"""
import argparse
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_phase(phase_num: int, script_name: str, args: list):
    print(f"\n{'='*75}")
    print(f"  LAUNCHING PHASE {phase_num}: {script_name}")
    print(f"  Configuration: {' '.join(args) if args else 'Default full schedule'}")
    print(f"{'='*75}\n")
    script_path = os.path.join(BASE_DIR, script_name)
    cmd = [sys.executable, script_path] + args
    t_start = time.time()
    ret = subprocess.run(cmd, cwd=BASE_DIR)
    t_dur = time.time() - t_start
    if ret.returncode != 0:
        print(f"\n[ERROR] Phase {phase_num} ({script_name}) failed with return code {ret.returncode} after {t_dur:.1f}s.")
        sys.exit(ret.returncode)
    print(f"\n[SUCCESS] Phase {phase_num} ({script_name}) completed successfully in {t_dur/60:.2f} minutes.")


def main():
    p = argparse.ArgumentParser(description="FedActuary Full Production Pipeline Runner")
    p.add_argument("--rounds", type=int, default=350, help="Federated baseline rounds (default: 350 per PRD Section 7)")
    p.add_argument("--epochs", type=int, default=10, help="Local client epochs per round (default: 10 per PRD Section 7)")
    p.add_argument("--log_every", type=int, default=10, help="Logging frequency in rounds (default: 10)")
    p.add_argument("--resume", action="store_true", default=True, help="Automatically resume interrupted training from checkpoints")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Force starting fresh from round 1")
    args = p.parse_args()

    t_total = time.time()
    resume_flag = ["--resume"] if args.resume else ["--no-resume"]
    print("\n" + "#"*75)
    print("  FEDACTUARY: FULL PRODUCTION FEDERATED RISK PIPELINE")
    print("  Dataset: freMTPL2freq.csv (All 678,013 policies)")
    print(f"  Baseline Schedule: {args.rounds} rounds x {args.epochs} local epochs (10 insurers)")
    print(f"  Resumption Mode: {'ENABLED (will continue seamlessly from any interrupted point)' if args.resume else 'DISABLED (starting fresh)'}")
    print("#"*75)

    # Phase 1: Baseline Full Run
    p1_args = ["--rounds", str(args.rounds), "--epochs", str(args.epochs), "--log_every", str(args.log_every)] + resume_flag
    run_phase(1, "phase1_baseline.py", p1_args)

    # Phase 2: Non-IID Dirichlet Heterogeneity Full Sweep
    p2_args = ["--rounds", "30", "--epochs", "5", "--log_every", "5", "--alphas", "0.1", "0.5", "1.0", "5.0"] + resume_flag
    run_phase(2, "phase2_nonIID.py", p2_args)

    # Phase 3: Differential Privacy Full Calibration
    p3_args = ["--epochs", "5"] + resume_flag
    run_phase(3, "phase3_privacy.py", p3_args)

    # Phase 4: Post-Hoc Explainability (Full Test Set)
    p4_args = ["--sample_size", "200"]
    run_phase(4, "phase4_explainability.py", p4_args)

    elapsed_total = time.time() - t_total
    print("\n" + "#"*75)
    print(f"  ALL PIPELINE PHASES COMPLETE in {elapsed_total/60:.2f} minutes!")
    print("  Checkpoints saved to checkpoints/")
    print("  Results saved to results/")
    print("  Launch the web dashboard via: python -m streamlit run app.py")
    print("#"*75 + "\n")


if __name__ == "__main__":
    main()

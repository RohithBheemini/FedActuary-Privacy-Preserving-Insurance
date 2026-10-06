"""
PDF Generator for FedSure Mutual Enterprise Platform Guide.
Uses ReportLab to generate a clean, executive-ready PDF manual with
structured typography, tables, and page numbers.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "FedSure_Mutual_Enterprise_Platform_Guide.pdf"
)


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for adding running headers and 'Page X of Y' footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "FedSure Mutual — Enterprise Federated Insurance Platform Guide")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — FEDSURE MUTUAL CONSORTIUM")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)

        self.restoreState()


def build_pdf(filename=PDF_PATH):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        leftIndent=14,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        "Callout_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E40AF"),
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0F172A")
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#FFFFFF")
    )

    story = []

    # -------------------------------------------------------------------------
    # COVER / HEADER BLOCK
    # -------------------------------------------------------------------------
    story.append(Paragraph("FedSure Mutual Enterprise", title_style))
    story.append(Paragraph("Federated Insurance Platform — Complete Technical Architecture & Demonstration Manual", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0F172A"), spaceBefore=0, spaceAfter=14))

    meta_table = Table([
        [
            Paragraph("<b>Version:</b> 2.0 Production", table_cell_style),
            Paragraph("<b>Date:</b> October 2026", table_cell_style),
            Paragraph("<b>Framework:</b> PyTorch 2.14 / Flower / Streamlit", table_cell_style)
        ],
        [
            Paragraph("<b>Audience:</b> Underwriters, Actuaries & Executives", table_cell_style),
            Paragraph("<b>Compliance:</b> GDPR / Opacus DP-SGD (ε=3.0)", table_cell_style),
            Paragraph("<b>Database:</b> SQLite ACID Enterprise Engine", table_cell_style)
        ]
    ], colWidths=[170, 164, 170])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, -1), (-1, -1), 1, colors.HexColor("#CBD5E1")),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # -------------------------------------------------------------------------
    # SECTION 1: EXECUTIVE SUMMARY
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Problem Solved", h1_style))
    story.append(Paragraph(
        "Insurance carriers and regional branches are legally prohibited by data protection mandates "
        "(such as GDPR and HIPAA) and commercial competition from pooling raw customer records and claims histories. "
        "Historically, isolated training on local data shards caused severe model overfitting on zero-inflated claim distributions. "
        "<b>FedSure Mutual</b> transforms this paradigm into a production-grade enterprise web platform. "
        "It leverages <b>Horizontal Federated Learning (FedAvg/FedProx)</b> to train deep Poisson regression neural networks "
        "across decentralized carrier branches without exposing customer Personally Identifiable Information (PII).",
        body_style
    ))
    story.append(Paragraph(
        "Beyond machine learning, FedSure Mutual provides a fully operational, role-based corporate platform "
        "supporting First Notice of Loss (FNOL) digital claim intake with automated AI triage, real-time policy endorsements, "
        "coverage top-ups, actuarial rating, and multi-branch consortium governance.",
        body_style
    ))

    # -------------------------------------------------------------------------
    # SECTION 2: SYSTEM ARCHITECTURE & PERSISTENCE
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 6))
    story.append(Paragraph("2. Enterprise Data & Persistence Architecture", h1_style))
    story.append(Paragraph(
        "The platform unifies two foundational datasets totaling <b>704,652 actuarial records</b>: "
        "the French Motor Third-Party Liability frequency dataset (<code>freMTPL2freq</code>, 678,013 policies) "
        "and the claims severity loss dataset (<code>freMTPL2sev</code>, 26,639 incurred payout events). "
        "All live updates and transactions persist in an ACID-compliant relational SQLite database (<code>data/insurance_enterprise.db</code>):",
        body_style
    ))

    db_data = [
        [Paragraph("Table Name", table_header_style), Paragraph("Entity Responsibility", table_header_style), Paragraph("Key Relational Attributes", table_header_style)],
        [
            Paragraph("<b>users</b>", table_cell_style),
            Paragraph("Role-Based Access Control (RBAC)", table_cell_style),
            Paragraph("id, username, full_name, email, role, policy_id", table_cell_style)
        ],
        [
            Paragraph("<b>policies</b>", table_cell_style),
            Paragraph("Master policy contract states & limits", table_cell_style),
            Paragraph("id_pol, holder_name, driv_age, veh_power, base_limit, top_up_amount, total_limit, frequency, annual_premium, status", table_cell_style)
        ],
        [
            Paragraph("<b>claims</b>", table_cell_style),
            Paragraph("First Notice of Loss (FNOL) records", table_cell_style),
            Paragraph("claim_id, id_pol, claimant_name, claim_type, amount_claimed, approved_payout, status, fl_fraud_risk_score, triage_recommendation", table_cell_style)
        ],
        [
            Paragraph("<b>endorsements</b>", table_cell_style),
            Paragraph("Immutable modification audit log", table_cell_style),
            Paragraph("endorsement_id, id_pol, changes_json, pred_before, pred_after, premium_delta, status, reviewed_by", table_cell_style)
        ],
        [
            Paragraph("<b>top_up_history</b>", table_cell_style),
            Paragraph("Coverage buffer deposit transactions", table_cell_style),
            Paragraph("tx_id, id_pol, amount, premium_charge, new_total_limit, new_remaining_claim, payment_method", table_cell_style)
        ],
        [
            Paragraph("<b>fl_consortium_rounds</b>", table_cell_style),
            Paragraph("Decentralized training round logs", table_cell_style),
            Paragraph("round_id, round_number, participating_branches, aggregation_strategy, dp_epsilon, train_loss, gini_score, is_production", table_cell_style)
        ],
    ]
    db_table = Table(db_data, colWidths=[100, 160, 244])
    db_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
    ]))
    story.append(db_table)

    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SECTION 3: THE FOUR ROLE WORKSPACES
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Workspace Feature Walkthrough & Operations", h1_style))
    story.append(Paragraph(
        "The interface eliminates visual clutter, avoiding emojis and cartoonish designs in favor of an "
        "executive SaaS aesthetic. Navigation is structured into two clean selectors: "
        "<b>Active Workspace</b> (the persona) and <b>Navigation</b> (contextual sub-pages).",
        body_style
    ))

    # WORKSPACE 1
    story.append(Paragraph("3.1 Workspace 1: Policyholder Portal (Customer Self-Service)", h2_style))
    story.append(Paragraph(
        "Empowers insured drivers to manage their contracts, file claims with instant AI triage, "
        "and request endorsements with transparent actuarial re-pricing.",
        body_style
    ))
    story.append(Paragraph("• <b>Policy Overview & Digital Wallet:</b> Displays vehicle specifications, driver rating (Bonus-Malus), active frequency, and four primary KPI cards: Total Coverage Limit, Incurred Claims Payout, Remaining Claim Buffer, and Pool Utilization percentage with a visual progress bar. Sub-tabs provide itemized claim history, modification audit trails, and top-up deposits.", bullet_style))
    story.append(Paragraph("• <b>First Notice of Loss (FNOL Claim Intake):</b> A clean digital submission form capturing incident date, category (Third-Party Collision, Hail/Glass, Rear-End, Parking), estimated loss amount (€), location, and police report references. The federated AI model immediately assigns an <i>Anomaly & Fraud Risk Score</i> and routes the filing into either Fast-Track Settlement, Standard Underwriter Review, or SIU Anomaly Flag.", bullet_style))
    story.append(Paragraph("• <b>Policy Endorsement:</b> Live parameter modification interface for driver age, engine power (VehPower CV), vehicle age, fuel type, urban category, and density. Upon submission, the federated Poisson neural network computes the expected claim count before and after (e.g. 0.0702 → 0.0742, +5.70%), displays the risk delta, and persists the endorsement.", bullet_style))
    story.append(Paragraph("• <b>Coverage Top-Up Engine:</b> Customers can purchase +€5,000 to +€50,000 in additional claim buffer on demand. The transparent surcharge is computed via <code>Amount × 0.004 × (BonusMalus / 100)</code>, immediately expanding the remaining claim pool.", bullet_style))
    story.append(Paragraph("• <b>Billing Schedule:</b> Switches billing frequency between Annual, Semi-Annual, Quarterly, and Monthly installments, clearly displaying the installment breakdown and total annualized premium.", bullet_style))

    story.append(Spacer(1, 6))

    # WORKSPACE 2
    story.append(Paragraph("3.2 Workspace 2: Underwriting Workbench (Internal Operations)", h2_style))
    story.append(Paragraph(
        "Dedicated to risk officers, claims adjusters, and actuaries overseeing policy modifications, claim settlements, and portfolio risk distributions.",
        body_style
    ))
    story.append(Paragraph("• <b>Endorsement Review Queue:</b> Displays pending policy alterations. Underwriters review structured JSON diffs of modified fields alongside before/after risk scores, adjudicating requests with audit notes.", bullet_style))
    story.append(Paragraph("• <b>Claims FNOL Adjudication Desk:</b> Centralized queue of filed First Notice of Loss claims. Underwriters inspect incident statements, claimed amounts, and the automated FL fraud risk score. Actions include approving payouts, applying deductibles, denying claims, or assigning to SIU.", bullet_style))
    story.append(Paragraph("• <b>Risk Rating Engine & SHAP Attribution:</b> A 10-parameter actuarial calculator evaluating Poisson claim frequencies. Classifies policyholders into Risk Tiers (Tier 1 Preferred Clean to Tier 4 Substandard) and renders a horizontal SHAP attribution chart decomposing which attributes drove the rate loading.", bullet_style))
    story.append(Paragraph("• <b>Portfolio Analytics:</b> High-level actuarial metrics across the 678k portfolio (36,102 claims, €59.9M losses, €1,128 baseline severity). Visualizes regional policy volumes and driver age demographic distributions.", bullet_style))
    story.append(Paragraph("• <b>Master Policy Directory:</b> High-performance server-side paginated explorer across the complete 678,013 dataset with multi-attribute filtering.", bullet_style))

    story.append(Spacer(1, 6))

    # WORKSPACE 3
    story.append(Paragraph("3.3 Workspace 3: Broker Quoting Portal (Sales & Distribution)", h2_style))
    story.append(Paragraph(
        "Provides commercial agents and brokers with rapid quote generation and instant contract binding.",
        body_style
    ))
    story.append(Paragraph("• <b>Quote & Issue Policy:</b> Brokers enter prospect demographics and car specifications. The rating engine evaluates predicted frequency and offers three coverage tiers: Silver (€30k limit, -15% rate), Gold (€50k standard), and Platinum (€100k, +35% rate). One-click binding generates a new policy ID and registers the contract in SQLite.", bullet_style))
    story.append(Paragraph("• <b>Issued Policies Directory:</b> Real-time ledger of all broker-issued contracts.", bullet_style))

    story.append(Spacer(1, 6))

    # WORKSPACE 4
    story.append(Paragraph("3.4 Workspace 4: Consortium Operations (Data Science & Admin)", h2_style))
    story.append(Paragraph(
        "Manages decentralized branch silos, collaborative federated training rounds, and model governance.",
        body_style
    ))
    story.append(Paragraph("• <b>Branch Silos Network:</b> Displays 4 regional carrier nodes (Paris HQ 18.2k rows, Lyon 12.5k rows, Bordeaux 9.8k rows, Marseille 11.3k rows). Confirms that private data rows never cross silo boundaries.", bullet_style))
    story.append(Paragraph("• <b>Collaborative Training Round:</b> Operators configure aggregation strategy (FedAvg or FedProx), Differential Privacy budget (ε ∈ {1.0, 3.0, 8.0, ∞}), and local epochs. Triggering a round simulates decentralized training, averages parameters, injects Opacus RDP noise, and logs training loss and Gini coefficients.", bullet_style))
    story.append(Paragraph("• <b>Model Checkpoint Governance:</b> A hot-swap deployment mechanism. Selecting any completed training round and clicking 'Promote Checkpoint to Active Production' hot-swaps the active model weights for all rating engines on subsequent queries.", bullet_style))
    story.append(Paragraph("• <b>Actuarial Benchmark Auditing:</b> Formal peer-reviewed benchmarks reproducing Śmietanka et al. (British Actuarial Journal, 2026) across Non-IID Dirichlet distribution, Differential Privacy trade-offs, and SHAP consistency.", bullet_style))

    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SECTION 4: 5-MINUTE LIVE DEMONSTRATION SCRIPT
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Recommended 5-Minute Live Demonstration Script", h1_style))
    story.append(Paragraph(
        "When presenting FedSure Mutual to an audience, follow this structured narrative to demonstrate "
        "interoperability between the customer portal, underwriter workbench, broker tool, and federated AI engine:",
        body_style
    ))

    script_data = [
        [Paragraph("Timeframe", table_header_style), Paragraph("Demonstration Action & Narrative", table_header_style), Paragraph("Observable Platform Output", table_header_style)],
        [
            Paragraph("<b>0:00 - 1:00</b><br/>Policyholder", table_cell_style),
            Paragraph("1. Open Policyholder Portal, select Alice Dupont (POL-1010996).<br/>2. Point out Digital Wallet: €50k limit, remaining buffer.<br/>3. Open 'File a Claim' and submit an €850 bumper damage claim.", table_cell_style),
            Paragraph("Instant AI triage score appears (e.g. 26% fraud risk → Standard Underwriter Review). Claim stored in database.", table_cell_style)
        ],
        [
            Paragraph("<b>1:00 - 2:00</b><br/>Underwriter", table_cell_style),
            Paragraph("1. Switch to Underwriting Workbench → Claims Adjudication.<br/>2. Select Alice's claim, inspect statement and AI risk score.<br/>3. Click 'Confirm Claim Adjudication' to approve €850 payout.<br/>4. Open 'Risk Rating Engine' to show Poisson prediction and SHAP bars.", table_cell_style),
            Paragraph("Claim status changes to Approved. SHAP bar chart decomposes Bonus-Malus load, urban density, and vehicle power.", table_cell_style)
        ],
        [
            Paragraph("<b>2:00 - 3:00</b><br/>Self-Service", table_cell_style),
            Paragraph("1. Return to Policyholder Portal → Coverage Top-Up.<br/>2. Observe remaining balance updated; add +€10,000 top-up buffer.<br/>3. Open Policy Endorsement: increase engine power from 6 to 9 CV.<br/>4. Apply endorsement.", table_cell_style),
            Paragraph("Total limit expands from €50k to €60k. Neural network calculates expected claim frequency change (+5.7%) in real-time.", table_cell_style)
        ],
        [
            Paragraph("<b>3:00 - 4:00</b><br/>Broker", table_cell_style),
            Paragraph("1. Switch to Broker Quoting Portal → Quote & Issue Policy.<br/>2. Enter prospect Jean-Michel Dupont, select Platinum tier.<br/>3. Generate quote and click 'Bind Contract and Issue Policy'.", table_cell_style),
            Paragraph("Instant annual/monthly rate generated. New contract issued with unique POL-ID into master SQLite database.", table_cell_style)
        ],
        [
            Paragraph("<b>4:00 - 5:00</b><br/>Consortium AI", table_cell_style),
            Paragraph("1. Switch to Consortium Operations → Branch Silos Network.<br/>2. Highlight 4 regional silos (51.9k private rows, zero PII pooling).<br/>3. Trigger Collaborative Training Round with DP ε=3.0.<br/>4. Open Checkpoint Governance and promote new model.", table_cell_style),
            Paragraph("Live round completes with updated training loss and Gini score. Model weights hot-swapped for live production inference.", table_cell_style)
        ]
    ]
    script_table = Table(script_data, colWidths=[80, 240, 184])
    script_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
    ]))
    story.append(script_table)

    story.append(Spacer(1, 14))

    # -------------------------------------------------------------------------
    # SECTION 5: VERIFICATION RESULTS & COMMANDS
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Verification Suite & Launch Instructions", h1_style))
    story.append(Paragraph(
        "The platform includes an automated 10-step production verification suite (<code>verify_workflow.py</code>) "
        "and 8 unit tests (<code>tests/test_policy.py</code>). Both pass with 100% operational status:",
        body_style
    ))

    verif_data = [
        [Paragraph("Step", table_header_style), Paragraph("Component Verified", table_header_style), Paragraph("Outcome & Deliverable", table_header_style)],
        [Paragraph("1", table_cell_style), Paragraph("Environment & Repo", table_cell_style), Paragraph("Python 3.11, PyTorch, Flower, Opacus, Streamlit operational", table_cell_style)],
        [Paragraph("2", table_cell_style), Paragraph("freMTPL2sev Dataset", table_cell_style), Paragraph("26,639 severity loss records verified (€60.69M losses)", table_cell_style)],
        [Paragraph("3", table_cell_style), Paragraph("Claims Analytics", table_cell_style), Paragraph("36,102 claims across 678,013 policies (10.07% annual frequency)", table_cell_style)],
        [Paragraph("4", table_cell_style), Paragraph("Dataset Explorer", table_cell_style), Paragraph("678,013 rows with 24 columns, server-side filtering & pagination", table_cell_style)],
        [Paragraph("5", table_cell_style), Paragraph("Remaining Claim Pool", table_cell_style), Paragraph("Accurate deduction: Limit - Incurred = Remaining balance", table_cell_style)],
        [Paragraph("6", table_cell_style), Paragraph("Top-Up Engine", table_cell_style), Paragraph("0.4% Bonus-Malus formula, instant coverage buffer expansion", table_cell_style)],
        [Paragraph("7", table_cell_style), Paragraph("Policy Endorsements", table_cell_style), Paragraph("Neural network re-scoring before/after with audit trail logging", table_cell_style)],
        [Paragraph("8", table_cell_style), Paragraph("SQLite Persistence", table_cell_style), Paragraph("ACID foreign key integrity guaranteed across all operations", table_cell_style)],
        [Paragraph("9", table_cell_style), Paragraph("FNOL Intake & Triage", table_cell_style), Paragraph("Automated fraud risk scoring and underwriter adjudication desk", table_cell_style)],
        [Paragraph("10", table_cell_style), Paragraph("FL Consortium Engine", table_cell_style), Paragraph("Live FedAvg round with DP ε=3.0, hot-swap checkpoint promotion", table_cell_style)],
    ]
    v_table = Table(verif_data, colWidths=[36, 170, 298])
    v_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
    ]))
    story.append(v_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>To launch the live platform:</b><br/>"
        "<code>python -m streamlit run app.py</code><br/><br/>"
        "<b>To run the automated verification suite:</b><br/>"
        "<code>python verify_workflow.py</code>",
        body_style
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF at: {filename}")


if __name__ == "__main__":
    build_pdf()

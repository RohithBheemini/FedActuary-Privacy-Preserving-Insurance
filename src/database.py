"""
Database Engine & Persistence Layer for FedSure Enterprise Insurance Platform.
Provides SQLite-backed persistence for Users, Policies, Endorsements, Claims (FNOL),
Top-up Transactions, and Federated Learning Consortium Training Logs.
"""

import json
import os
import sqlite3
import datetime
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "insurance_enterprise.db")


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite database with row factory for dict-like access."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(force_reinit: bool = False):
    """
    Initializes all database tables. If force_reinit is True or DB is empty,
    creates the schema and seeds baseline users, sample policies, and active claims.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    if force_reinit:
        cursor.execute("DROP TABLE IF EXISTS fl_consortium_rounds;")
        cursor.execute("DROP TABLE IF EXISTS top_up_history;")
        cursor.execute("DROP TABLE IF EXISTS claims;")
        cursor.execute("DROP TABLE IF EXISTS endorsements;")
        cursor.execute("DROP TABLE IF EXISTS policies;")
        cursor.execute("DROP TABLE IF EXISTS users;")

    # 1. Users Table (Role-Based Access Control)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('policyholder', 'underwriter', 'agent', 'admin')),
        policy_id INTEGER,
        created_at TEXT NOT NULL
    );
    """)

    # 2. Policies Table (Master Record with Actuarial & FL Features)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS policies (
        id_pol INTEGER PRIMARY KEY,
        holder_name TEXT NOT NULL,
        holder_email TEXT NOT NULL,
        driv_age INTEGER NOT NULL,
        bonus_malus INTEGER NOT NULL,
        exposure REAL NOT NULL DEFAULT 1.0,
        veh_power INTEGER NOT NULL,
        veh_age INTEGER NOT NULL,
        veh_gas TEXT NOT NULL,
        veh_brand TEXT NOT NULL,
        area TEXT NOT NULL,
        region TEXT NOT NULL,
        density REAL NOT NULL,
        base_limit REAL NOT NULL DEFAULT 50000.0,
        top_up_amount REAL NOT NULL DEFAULT 0.0,
        total_limit REAL NOT NULL DEFAULT 50000.0,
        frequency TEXT NOT NULL DEFAULT 'Annual',
        annual_premium REAL NOT NULL DEFAULT 250.0,
        status TEXT NOT NULL DEFAULT 'Active (Clean)',
        fl_predicted_claim_freq REAL DEFAULT 0.085,
        fl_risk_tier TEXT DEFAULT 'Standard Risk',
        version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    # 3. Endorsements Table (Audit Trail for Policy Modifications)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS endorsements (
        endorsement_id TEXT PRIMARY KEY,
        id_pol INTEGER NOT NULL,
        requested_by TEXT NOT NULL,
        requested_at TEXT NOT NULL,
        endorsement_type TEXT NOT NULL,
        changes_json TEXT NOT NULL,
        pred_before REAL,
        pred_after REAL,
        premium_delta REAL DEFAULT 0.0,
        status TEXT NOT NULL CHECK(status IN ('Pending Review', 'Approved', 'Rejected', 'Auto-Approved')),
        notes TEXT,
        reviewed_by TEXT,
        reviewed_at TEXT,
        FOREIGN KEY (id_pol) REFERENCES policies(id_pol) ON DELETE CASCADE
    );
    """)

    # 4. Claims (FNOL - First Notice of Loss)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS claims (
        claim_id TEXT PRIMARY KEY,
        id_pol INTEGER NOT NULL,
        claimant_name TEXT NOT NULL,
        incident_date TEXT NOT NULL,
        filing_date TEXT NOT NULL,
        claim_type TEXT NOT NULL,
        amount_claimed REAL NOT NULL,
        approved_payout REAL DEFAULT 0.0,
        status TEXT NOT NULL CHECK(status IN ('Submitted', 'AI Triaged', 'Under Review', 'Approved', 'Settled', 'Denied')),
        description TEXT NOT NULL,
        fl_fraud_risk_score REAL DEFAULT 0.15,
        triage_recommendation TEXT DEFAULT 'Standard Fast-Track',
        underwriter_notes TEXT,
        evidence_attachment TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (id_pol) REFERENCES policies(id_pol) ON DELETE CASCADE
    );
    """)

    # 5. Top-Up Transactions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS top_up_history (
        tx_id TEXT PRIMARY KEY,
        id_pol INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        amount REAL NOT NULL,
        premium_charge REAL NOT NULL,
        new_total_limit REAL NOT NULL,
        new_remaining_claim REAL NOT NULL,
        payment_method TEXT DEFAULT 'Credit Card',
        status TEXT DEFAULT 'Completed',
        FOREIGN KEY (id_pol) REFERENCES policies(id_pol) ON DELETE CASCADE
    );
    """)

    # 6. Federated Learning Consortium Rounds
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fl_consortium_rounds (
        round_id TEXT PRIMARY KEY,
        consortium_name TEXT NOT NULL,
        round_number INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        participating_branches TEXT NOT NULL,
        aggregation_strategy TEXT NOT NULL,
        dp_epsilon REAL,
        train_loss REAL NOT NULL,
        pde_score REAL,
        gini_score REAL,
        model_checkpoint_path TEXT NOT NULL,
        is_production INTEGER NOT NULL DEFAULT 0
    );
    """)

    conn.commit()

    # Seed initial data if tables are newly created
    cursor.execute("SELECT COUNT(*) FROM users;")
    user_count = cursor.fetchone()[0]
    if user_count == 0:
        _seed_initial_data(conn)

    conn.close()


def _seed_initial_data(conn: sqlite3.Connection):
    """Populates initial realistic demo users, sample policies, active claims, and FL rounds."""
    cursor = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Seed Roles
    users = [
        ("alice", "Alice Dupont", "alice.dupont@fedsure-demo.fr", "policyholder", 1010996, now_str),
        ("marc", "Marc Leroy", "marc.leroy@fedsure-demo.fr", "policyholder", 1552, now_str),
        ("claire", "Claire Moreau", "claire.moreau@fedsure-demo.fr", "policyholder", 424, now_str),
        ("sarah_underwriter", "Sarah Jenkins (Lead Underwriter)", "s.jenkins@fedsure-group.com", "underwriter", None, now_str),
        ("jean_agent", "Jean-Paul Belmondo (Regional Broker)", "jp.agent@fedsure-brokers.com", "agent", None, now_str),
        ("admin_fl", "Dr. Alexandre Vane (FL Consortium Director)", "a.vane@fedsure-consortium.org", "admin", None, now_str),
    ]
    cursor.executemany("""
    INSERT INTO users (username, full_name, email, role, policy_id, created_at)
    VALUES (?, ?, ?, ?, ?, ?);
    """, users)

    # 2. Seed Real Policies (Matched with freMTPL2 sample IDs)
    sample_policies = [
        (
            1010996, "Alice Dupont", "alice.dupont@fedsure-demo.fr",
            34, 50, 0.95, 6, 4, "Regular", "B12", "D", "R82", 1217.0,
            50000.0, 0.0, 50000.0, "Annual", 285.50, "Active with Claims",
            0.078, "Low Risk", 1, now_str, now_str
        ),
        (
            1552, "Marc Leroy", "marc.leroy@fedsure-demo.fr",
            48, 65, 1.0, 7, 6, "Diesel", "B3", "C", "R24", 845.0,
            50000.0, 15000.0, 65000.0, "Monthly", 340.00, "Active with Claims",
            0.092, "Standard Risk", 2, now_str, now_str
        ),
        (
            424, "Claire Moreau", "claire.moreau@fedsure-demo.fr",
            22, 100, 0.85, 9, 2, "Regular", "B1", "E", "R11", 3450.0,
            50000.0, 0.0, 50000.0, "Quarterly", 495.00, "Active (Clean)",
            0.185, "High Risk", 1, now_str, now_str
        ),
        (
            202601, "Guillaume Bernard", "g.bernard@example.fr",
            52, 50, 1.0, 5, 8, "Diesel", "B2", "B", "R53", 420.0,
            60000.0, 0.0, 60000.0, "Semi-Annual", 220.00, "Active (Clean)",
            0.052, "Low Risk", 1, now_str, now_str
        ),
        (
            202602, "Sophie Martin", "s.martin@example.fr",
            29, 85, 0.9, 8, 3, "Regular", "B14", "D", "R72", 1100.0,
            50000.0, 0.0, 50000.0, "Annual", 380.00, "Active with Claims",
            0.124, "Elevated Risk", 1, now_str, now_str
        )
    ]
    cursor.executemany("""
    INSERT INTO policies (
        id_pol, holder_name, holder_email, driv_age, bonus_malus, exposure,
        veh_power, veh_age, veh_gas, veh_brand, area, region, density,
        base_limit, top_up_amount, total_limit, frequency, annual_premium,
        status, fl_predicted_claim_freq, fl_risk_tier, version, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, sample_policies)

    # 3. Seed Claims (FNOL)
    claims = [
        (
            "CLM-2026-0081", 1010996, "Alice Dupont", "2026-08-14", "2026-08-15",
            "Third-Party Collision (Intersection)", 1128.12, 1128.12, "Settled",
            "Minor fender bender during evening commute at Rue de la Paix. Police report filed.",
            0.12, "Auto Fast-Track (Low Risk)", "Approved and paid out per standard liability schedule.",
            None, now_str
        ),
        (
            "CLM-2026-0094", 1552, "Marc Leroy", "2026-09-02", "2026-09-03",
            "Windshield Glass & Hail Damage", 995.20, 995.20, "Approved",
            "Severe hailstorm shattered rear passenger window and cracked front windscreen.",
            0.08, "Auto Fast-Track (Low Risk)", "Photographic evidence validated by automated vision triage.",
            None, now_str
        ),
        (
            "CLM-2026-0105", 202602, "Sophie Martin", "2026-10-01", "2026-10-02",
            "Rear-End Impact on Highway A6", 4250.00, 0.0, "Under Review",
            "Sudden brake scenario on highway off-ramp. Significant bumper and radiator damage.",
            0.45, "Underwriter Adjudication Recommended", "Assigned to Sarah Jenkins for repair shop estimate audit.",
            None, now_str
        )
    ]
    cursor.executemany("""
    INSERT INTO claims (
        claim_id, id_pol, claimant_name, incident_date, filing_date, claim_type,
        amount_claimed, approved_payout, status, description, fl_fraud_risk_score,
        triage_recommendation, underwriter_notes, evidence_attachment, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, claims)

    # 4. Seed Endorsements (Audit Trail)
    endorsements = [
        (
            "END-2026-0012", 1552, "Marc Leroy", "2026-07-10 14:20:00",
            "Coverage Top-Up & Billing Cycle Change",
            json.dumps({"old": {"frequency": "Annual", "total_limit": 50000.0}, "new": {"frequency": "Monthly", "total_limit": 65000.0}}),
            0.092, 0.092, 45.00, "Approved", "Customer upgraded coverage buffer before cross-country family road trip.",
            "Sarah Jenkins", "2026-07-10 15:00:00"
        ),
        (
            "END-2026-0018", 1010996, "Alice Dupont", "2026-09-20 11:05:00",
            "Vehicle Spec Update (VehAge increment)",
            json.dumps({"old": {"veh_age": 3}, "new": {"veh_age": 4}}),
            0.081, 0.078, -12.50, "Auto-Approved", "Annual automated policy vehicle age depreciation adjustment.",
            "System Bot", "2026-09-20 11:05:01"
        )
    ]
    cursor.executemany("""
    INSERT INTO endorsements (
        endorsement_id, id_pol, requested_by, requested_at, endorsement_type,
        changes_json, pred_before, pred_after, premium_delta, status, notes,
        reviewed_by, reviewed_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, endorsements)

    # 5. Seed Top-up History
    top_ups = [
        (
            "TOP-2026-8801", 1552, "2026-07-10 14:22:15",
            15000.0, 75.0, 65000.0, 64004.80, "Corporate Direct Debit", "Completed"
        )
    ]
    cursor.executemany("""
    INSERT INTO top_up_history (
        tx_id, id_pol, timestamp, amount, premium_charge, new_total_limit,
        new_remaining_claim, payment_method, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, top_ups)

    # 6. Seed FL Consortium Production Training Rounds
    fl_rounds = [
        (
            "FL-RND-001", "Pan-European Insurance Privacy Consortium", 10,
            "2026-09-01 03:00:00", "Paris HQ, Lyon Branch, Marseille Silo, Lille Agency",
            "FedAvg", None, 0.2391, 0.244, -0.672, "checkpoints/federated.pt", 0
        ),
        (
            "FL-RND-002", "Pan-European Insurance Privacy Consortium", 30,
            "2026-09-15 03:00:00", "Paris HQ, Lyon Branch, Marseille Silo, Bordeaux Hub",
            "FedAvg + Client NAdam", 3.0, 0.2224, 0.253, -0.869, "checkpoints/dp_eps_3.0.pt", 0
        ),
        (
            "FL-RND-003", "Pan-European Insurance Privacy Consortium", 50,
            "2026-10-01 04:30:00", "All 10 Regional French Silos (Decentralized)",
            "FedAvg + Differential Privacy (RDP)", 3.0, 0.2123, 0.265, -0.259, "checkpoints/federated.pt", 1
        )
    ]
    cursor.executemany("""
    INSERT INTO fl_consortium_rounds (
        round_id, consortium_name, round_number, timestamp, participating_branches,
        aggregation_strategy, dp_epsilon, train_loss, gini_score, pde_score,
        model_checkpoint_path, is_production
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, fl_rounds)

    conn.commit()


# -----------------------------------------------------------------------------
# Database Access Helpers
# -----------------------------------------------------------------------------

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM users WHERE username = ?;", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_users() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM users ORDER BY id ASC;").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_policy_from_db(id_pol: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM policies WHERE id_pol = ?;", (id_pol,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_policies_from_db() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM policies ORDER BY updated_at DESC;").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def upsert_policy_in_db(policy_data: Dict[str, Any]) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    id_pol = int(policy_data["id_pol"])

    existing = cursor.execute("SELECT id_pol FROM policies WHERE id_pol = ?;", (id_pol,)).fetchone()
    if existing:
        cursor.execute("""
        UPDATE policies SET
            holder_name = COALESCE(?, holder_name),
            holder_email = COALESCE(?, holder_email),
            driv_age = COALESCE(?, driv_age),
            bonus_malus = COALESCE(?, bonus_malus),
            exposure = COALESCE(?, exposure),
            veh_power = COALESCE(?, veh_power),
            veh_age = COALESCE(?, veh_age),
            veh_gas = COALESCE(?, veh_gas),
            veh_brand = COALESCE(?, veh_brand),
            area = COALESCE(?, area),
            region = COALESCE(?, region),
            density = COALESCE(?, density),
            base_limit = COALESCE(?, base_limit),
            top_up_amount = COALESCE(?, top_up_amount),
            total_limit = COALESCE(?, total_limit),
            frequency = COALESCE(?, frequency),
            annual_premium = COALESCE(?, annual_premium),
            status = COALESCE(?, status),
            fl_predicted_claim_freq = COALESCE(?, fl_predicted_claim_freq),
            fl_risk_tier = COALESCE(?, fl_risk_tier),
            version = version + 1,
            updated_at = ?
        WHERE id_pol = ?;
        """, (
            policy_data.get("holder_name"),
            policy_data.get("holder_email"),
            policy_data.get("driv_age"),
            policy_data.get("bonus_malus"),
            policy_data.get("exposure"),
            policy_data.get("veh_power"),
            policy_data.get("veh_age"),
            policy_data.get("veh_gas"),
            policy_data.get("veh_brand"),
            policy_data.get("area"),
            policy_data.get("region"),
            policy_data.get("density"),
            policy_data.get("base_limit"),
            policy_data.get("top_up_amount"),
            policy_data.get("total_limit"),
            policy_data.get("frequency"),
            policy_data.get("annual_premium"),
            policy_data.get("status"),
            policy_data.get("fl_predicted_claim_freq"),
            policy_data.get("fl_risk_tier"),
            now_str,
            id_pol
        ))
    else:
        cursor.execute("""
        INSERT INTO policies (
            id_pol, holder_name, holder_email, driv_age, bonus_malus, exposure,
            veh_power, veh_age, veh_gas, veh_brand, area, region, density,
            base_limit, top_up_amount, total_limit, frequency, annual_premium,
            status, fl_predicted_claim_freq, fl_risk_tier, version, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?);
        """, (
            id_pol,
            policy_data.get("holder_name", f"Policyholder #{id_pol}"),
            policy_data.get("holder_email", f"client_{id_pol}@fedsure.fr"),
            policy_data.get("driv_age", 40),
            policy_data.get("bonus_malus", 50),
            policy_data.get("exposure", 1.0),
            policy_data.get("veh_power", 6),
            policy_data.get("veh_age", 5),
            policy_data.get("veh_gas", "Regular"),
            policy_data.get("veh_brand", "B12"),
            policy_data.get("area", "C"),
            policy_data.get("region", "R82"),
            policy_data.get("density", 1000.0),
            policy_data.get("base_limit", 50000.0),
            policy_data.get("top_up_amount", 0.0),
            policy_data.get("total_limit", 50000.0),
            policy_data.get("frequency", "Annual"),
            policy_data.get("annual_premium", 250.0),
            policy_data.get("status", "Active (Clean)"),
            policy_data.get("fl_predicted_claim_freq", 0.08),
            policy_data.get("fl_risk_tier", "Standard Risk"),
            now_str,
            now_str
        ))
    conn.commit()
    conn.close()


def _ensure_policy_exists(cursor, id_pol: int, holder_name: Optional[str] = None):
    """Ensures parent policy row exists in policies table to satisfy foreign keys."""
    existing = cursor.execute("SELECT id_pol FROM policies WHERE id_pol = ?;", (id_pol,)).fetchone()
    if not existing:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT INTO policies (
            id_pol, holder_name, holder_email, driv_age, bonus_malus, exposure,
            veh_power, veh_age, veh_gas, veh_brand, area, region, density,
            base_limit, top_up_amount, total_limit, frequency, annual_premium,
            status, fl_predicted_claim_freq, fl_risk_tier, version, created_at, updated_at
        ) VALUES (?, ?, ?, 35, 50, 1.0, 6, 4, 'Regular', 'B12', 'C', 'R82', 1000.0,
                  50000.0, 0.0, 50000.0, 'Annual', 250.0, 'Active with Claims', 0.08, 'Standard Risk', 1, ?, ?);
        """, (
            id_pol,
            holder_name or f"Policyholder #{id_pol}",
            f"client_{id_pol}@fedsure.fr",
            now_str,
            now_str
        ))


def add_endorsement(
    id_pol: int,
    requested_by: str,
    endorsement_type: str,
    changes_json: str,
    pred_before: Optional[float] = None,
    pred_after: Optional[float] = None,
    premium_delta: float = 0.0,
    status: str = "Pending Review",
    notes: str = "",
    reviewed_by: Optional[str] = None
) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_policy_exists(cursor, id_pol, requested_by)
    endorsement_id = f"END-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{int(datetime.datetime.now().microsecond / 1000)}"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO endorsements (
        endorsement_id, id_pol, requested_by, requested_at, endorsement_type,
        changes_json, pred_before, pred_after, premium_delta, status, notes,
        reviewed_by, reviewed_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        endorsement_id, id_pol, requested_by, now_str, endorsement_type,
        changes_json, pred_before, pred_after, premium_delta, status, notes,
        reviewed_by, now_str if reviewed_by else None
    ))
    conn.commit()
    conn.close()
    return endorsement_id


def get_endorsements_for_policy(id_pol: int) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM endorsements WHERE id_pol = ? ORDER BY requested_at DESC;", (id_pol,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_endorsements(status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    if status:
        rows = conn.execute("SELECT * FROM endorsements WHERE status = ? ORDER BY requested_at DESC;", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM endorsements ORDER BY requested_at DESC;").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_endorsement_status(endorsement_id: str, new_status: str, reviewed_by: str, notes: Optional[str] = None) -> None:
    conn = get_db_connection()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
    UPDATE endorsements SET
        status = ?,
        reviewed_by = ?,
        reviewed_at = ?,
        notes = COALESCE(?, notes)
    WHERE endorsement_id = ?;
    """, (new_status, reviewed_by, now_str, notes, endorsement_id))
    conn.commit()
    conn.close()


def add_claim(
    id_pol: int,
    claimant_name: str,
    incident_date: str,
    claim_type: str,
    amount_claimed: float,
    description: str,
    fl_fraud_risk_score: float = 0.15,
    triage_recommendation: str = "Standard Fast-Track",
    evidence_attachment: Optional[str] = None
) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_policy_exists(cursor, id_pol, claimant_name)
    claim_id = f"CLM-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{int(datetime.datetime.now().microsecond / 1000)}"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO claims (
        claim_id, id_pol, claimant_name, incident_date, filing_date, claim_type,
        amount_claimed, approved_payout, status, description, fl_fraud_risk_score,
        triage_recommendation, underwriter_notes, evidence_attachment, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, 'Submitted', ?, ?, ?, NULL, ?, ?);
    """, (
        claim_id, id_pol, claimant_name, incident_date, now_str[:10], claim_type,
        amount_claimed, description, fl_fraud_risk_score, triage_recommendation,
        evidence_attachment, now_str
    ))
    conn.commit()
    conn.close()
    return claim_id


def get_claims_for_policy(id_pol: int) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM claims WHERE id_pol = ? ORDER BY filing_date DESC;", (id_pol,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_claims(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    if status_filter and status_filter != "All":
        rows = conn.execute("SELECT * FROM claims WHERE status = ? ORDER BY filing_date DESC;", (status_filter,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM claims ORDER BY filing_date DESC;").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_claim_adjudication(
    claim_id: str,
    new_status: str,
    approved_payout: float,
    underwriter_notes: str
) -> None:
    conn = get_db_connection()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
    UPDATE claims SET
        status = ?,
        approved_payout = ?,
        underwriter_notes = ?,
        updated_at = ?
    WHERE claim_id = ?;
    """, (new_status, approved_payout, underwriter_notes, now_str, claim_id))
    conn.commit()
    conn.close()


def add_top_up_tx(
    id_pol: int,
    amount: float,
    premium_charge: float,
    new_total_limit: float,
    new_remaining_claim: float,
    payment_method: str = "Credit Card"
) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    _ensure_policy_exists(cursor, id_pol)
    tx_id = f"TOP-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{int(datetime.datetime.now().microsecond / 1000)}"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO top_up_history (
        tx_id, id_pol, timestamp, amount, premium_charge, new_total_limit,
        new_remaining_claim, payment_method, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Completed');
    """, (
        tx_id, id_pol, now_str, amount, premium_charge, new_total_limit,
        new_remaining_claim, payment_method
    ))

    # Also update policy record directly
    cursor.execute("""
    UPDATE policies SET
        top_up_amount = top_up_amount + ?,
        total_limit = ?,
        updated_at = ?
    WHERE id_pol = ?;
    """, (amount, new_total_limit, now_str, id_pol))

    conn.commit()
    conn.close()
    return tx_id


def get_top_up_history(id_pol: int) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM top_up_history WHERE id_pol = ? ORDER BY timestamp DESC;", (id_pol,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_fl_consortium_rounds() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM fl_consortium_rounds ORDER BY round_number DESC;").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_fl_round(
    consortium_name: str,
    round_number: int,
    participating_branches: str,
    aggregation_strategy: str,
    dp_epsilon: Optional[float],
    train_loss: float,
    pde_score: Optional[float],
    gini_score: Optional[float],
    model_checkpoint_path: str,
    is_production: bool = False
) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    round_id = f"FL-RND-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{int(datetime.datetime.now().microsecond / 1000)}"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if is_production:
        cursor.execute("UPDATE fl_consortium_rounds SET is_production = 0;")

    cursor.execute("""
    INSERT INTO fl_consortium_rounds (
        round_id, consortium_name, round_number, timestamp, participating_branches,
        aggregation_strategy, dp_epsilon, train_loss, pde_score, gini_score,
        model_checkpoint_path, is_production
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        round_id, consortium_name, round_number, now_str, participating_branches,
        aggregation_strategy, dp_epsilon, train_loss, pde_score, gini_score,
        model_checkpoint_path, 1 if is_production else 0
    ))
    conn.commit()
    conn.close()
    return round_id


def set_production_checkpoint(round_id_or_path: str) -> bool:
    conn = get_db_connection()
    conn.execute("UPDATE fl_consortium_rounds SET is_production = 0;")
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE fl_consortium_rounds
    SET is_production = 1
    WHERE round_id = ? OR model_checkpoint_path = ?;
    """, (round_id_or_path, round_id_or_path))
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated

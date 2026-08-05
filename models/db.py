"""
PayGuard persistence layer  (SQLite — Python stdlib, no extra dependency)
=========================================================================
Gives PayGuard durable storage so the "learned" state survives an API restart:
  - reputation ledger  (beneficiary / device risk — the fraud memory)
  - customer trust      (Dynamic Trust Engine per-customer scores)
  - transactions        (the live feed / history)

This replaces the previous in-memory-only stores. The DB file lives at
data/payguard.db and is created automatically on first run.

(For a large production deployment the PRD calls for PostgreSQL; SQLite gives us
the same durability with zero setup, which is right for this stage.)
"""

import sqlite3, threading
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "payguard.db"


class PayGuardDB:
    def __init__(self, path=DB_PATH):
        self.path = str(path)
        self._lock = threading.Lock()          # serialise writes (FastAPI is threaded)
        self._init()

    def _conn(self):
        c = sqlite3.connect(self.path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init(self):
        with self._lock, self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS reputation(
                entity_id TEXT PRIMARY KEY, kind TEXT, risk REAL, fraud_reports INTEGER,
                times_seen INTEGER, first_seen REAL, last_updated REAL, note TEXT, source TEXT);
            CREATE TABLE IF NOT EXISTS customers(
                customer_id TEXT PRIMARY KEY, trust_score REAL, baseline_trust REAL,
                total_transactions INTEGER, fraud_flags INTEGER, consecutive_normal INTEGER,
                last_transaction_time REAL);
            CREATE TABLE IF NOT EXISTS transactions(
                transaction_id TEXT PRIMARY KEY, ts TEXT, channel TEXT, amount REAL,
                customer_id TEXT, beneficiary_id TEXT, beneficiary_name TEXT, tti REAL,
                decision TEXT, fraud_probability REAL, trust_score REAL, top_reason TEXT,
                detail_json TEXT);
            """)
            # Migration for databases created before detail_json existed.
            try:
                c.execute("ALTER TABLE transactions ADD COLUMN detail_json TEXT")
            except sqlite3.OperationalError:
                pass  # column already present

    # ── Reputation ledger ──
    def load_reputation(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM reputation")]

    def upsert_reputation(self, r: dict):
        with self._lock, self._conn() as c:
            c.execute("""INSERT INTO reputation
                (entity_id,kind,risk,fraud_reports,times_seen,first_seen,last_updated,note,source)
                VALUES(:entity_id,:kind,:risk,:fraud_reports,:times_seen,:first_seen,:last_updated,:note,:source)
                ON CONFLICT(entity_id) DO UPDATE SET
                kind=excluded.kind, risk=excluded.risk, fraud_reports=excluded.fraud_reports,
                times_seen=excluded.times_seen, last_updated=excluded.last_updated,
                note=excluded.note, source=excluded.source""",
                {"entity_id": r["entity_id"], "kind": r["kind"], "risk": r["risk"],
                 "fraud_reports": r.get("fraud_reports", 0), "times_seen": r.get("times_seen", 0),
                 "first_seen": r.get("first_seen", 0), "last_updated": r.get("last_updated", 0),
                 "note": r.get("note", ""), "source": r.get("source", "runtime")})

    # ── Customers (trust) ──
    def load_customers(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM customers")]

    def upsert_customer(self, r: dict):
        with self._lock, self._conn() as c:
            c.execute("""INSERT INTO customers
                (customer_id,trust_score,baseline_trust,total_transactions,fraud_flags,consecutive_normal,last_transaction_time)
                VALUES(:customer_id,:trust_score,:baseline_trust,:total_transactions,:fraud_flags,:consecutive_normal,:last_transaction_time)
                ON CONFLICT(customer_id) DO UPDATE SET
                trust_score=excluded.trust_score, total_transactions=excluded.total_transactions,
                fraud_flags=excluded.fraud_flags, consecutive_normal=excluded.consecutive_normal,
                last_transaction_time=excluded.last_transaction_time""", r)

    # ── Transactions (feed / history / investigation) ──
    # Summary columns returned for lists (detail_json is fetched separately).
    _SUMMARY = ("transaction_id", "ts", "channel", "amount", "customer_id",
                "beneficiary_id", "beneficiary_name", "tti", "decision",
                "fraud_probability", "trust_score", "top_reason")

    def insert_transaction(self, e: dict, detail: dict | None = None):
        import json
        with self._lock, self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO transactions
                (transaction_id,ts,channel,amount,customer_id,beneficiary_id,beneficiary_name,
                 tti,decision,fraud_probability,trust_score,top_reason,detail_json)
                VALUES(:transaction_id,:ts,:channel,:amount,:customer_id,:beneficiary_id,:beneficiary_name,
                 :tti,:decision,:fraud_probability,:trust_score,:top_reason,:detail_json)""",
                {"transaction_id": e["transaction_id"], "ts": e.get("timestamp"),
                 "channel": e.get("channel"), "amount": e.get("amount"),
                 "customer_id": e.get("customer_id"), "beneficiary_id": e.get("beneficiary_id"),
                 "beneficiary_name": e.get("beneficiary_name"), "tti": e.get("tti"),
                 "decision": e.get("decision"), "fraud_probability": e.get("fraud_probability"),
                 "trust_score": e.get("trust_score"), "top_reason": e.get("top_reason"),
                 "detail_json": json.dumps(detail) if detail else None})

    def _summary_rows(self, limit) -> list[dict]:
        cols = ",".join(self._SUMMARY)
        with self._conn() as c:
            rows = c.execute(f"SELECT {cols} FROM transactions ORDER BY ts DESC LIMIT ?", (limit,))
            out = []
            for r in rows:
                d = dict(r)
                d["timestamp"] = d.pop("ts")     # match the feed's key name
                out.append(d)
            return out

    def recent_transactions(self, limit=100) -> list[dict]:
        return self._summary_rows(limit)

    def list_transactions(self, limit=300) -> list[dict]:
        return self._summary_rows(limit)

    def get_transaction(self, transaction_id: str) -> dict | None:
        import json
        with self._conn() as c:
            r = c.execute("SELECT * FROM transactions WHERE transaction_id=?", (transaction_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        detail = d.pop("detail_json", None)
        return {"summary": d, "detail": json.loads(detail) if detail else None}

    def decision_counts(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT decision, COUNT(*) n FROM transactions GROUP BY decision")
            m = {r["decision"]: r["n"] for r in rows}
        return {"total": sum(m.values()),
                "block": m.get("BLOCK", 0), "verify": m.get("STEP_UP_MFA", 0),
                "approve": m.get("APPROVE", 0)}

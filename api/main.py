"""
Continuous Trust Intelligence Framework — FastAPI Backend
Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import os, sys, time, uuid, hashlib, math
from datetime import datetime
from collections import deque
from typing import Optional, List
from pathlib import Path

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add parent to path for trust engine import
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
from trust_engine import DynamicTrustEngine
from reputation import ReputationStore
from db import PaymentGuardianDB

app = FastAPI(title="PaymentGuardian API", version="2.0.0",
              description="Real-time payment fraud protection — Risk Score (6-check blend) + LightGBM + SHAP")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = Path(__file__).parent.parent / "models"
db = PaymentGuardianDB()                                  # SQLite persistence (data/paymentguardian.db)
trust_engine = DynamicTrustEngine(db=db)
reputation_store = ReputationStore(db=db)
DEFAULT_STARTING_BALANCE = 184500  # matches the users.balance column default in models/db.py

# ── IOB Pay "pay to" contacts — seeded once, split roughly evenly across
# safe / moderate / risky so the app has something appealing (and instructive)
# to demo on first run. The two mule accounts double as entries already
# flagged in the reputation ledger (models/reputation.py), so paying them
# exercises the ledger-override path too. ──
DEFAULT_BENEFICIARIES = [
    {"beneficiary_id": "BENF-TRUSTED-01", "name": "Priya Sharma",        "bank_label": "HDFC Bank ••1234",   "tier": "safe",     "risk_score": 8,  "account_age_days": 620, "is_new": 0, "note": "Saved payee · 3 yrs",        "sort_order": 1},
    {"beneficiary_id": "BENF-TRUSTED-02", "name": "Rohan Mehta",         "bank_label": "ICICI Bank ••5678",  "tier": "safe",     "risk_score": 6,  "account_age_days": 900, "is_new": 0, "note": "Saved payee · frequent",     "sort_order": 2},
    {"beneficiary_id": "BENF-TRUSTED-03", "name": "Anita Desai",         "bank_label": "SBI ••2211",         "tier": "safe",     "risk_score": 10, "account_age_days": 410, "is_new": 0, "note": "Landlord · monthly rent",    "sort_order": 3},
    {"beneficiary_id": "BENF-TRUSTED-04", "name": "Amazon Pay",          "bank_label": "Axis Bank ••7788",   "tier": "safe",     "risk_score": 12, "account_age_days": 730, "is_new": 0, "note": "Verified merchant",          "sort_order": 4},
    {"beneficiary_id": "BENF-NEW-QM01",   "name": "QuickMart Online",    "bank_label": "Kotak ••4432",       "tier": "moderate", "risk_score": 40, "account_age_days": 45,  "is_new": 1, "note": "First-time payee",           "sort_order": 5},
    {"beneficiary_id": "BENF-NEW-VS02",   "name": "Vikram Singh",        "bank_label": "Yes Bank ••9090",    "tier": "moderate", "risk_score": 48, "account_age_days": 20,  "is_new": 1, "note": "Added this week",            "sort_order": 6},
    {"beneficiary_id": "BENF-NEW-NK03",   "name": "Neha Kulkarni",       "bank_label": "IDFC First ••3345",  "tier": "moderate", "risk_score": 52, "account_age_days": 12,  "is_new": 1, "note": "Friend · unsaved",           "sort_order": 7},
    {"beneficiary_id": "BENF-MULE-001",   "name": "Unknown A/C ••9021",  "bank_label": "Unverified bank",    "tier": "risky",    "risk_score": 92, "account_age_days": 3,   "is_new": 1, "note": "⚠ Flagged: known mule",      "sort_order": 8},
    {"beneficiary_id": "BENF-MULE-002",   "name": "Fast Cash Traders",   "bank_label": "Unverified bank",    "tier": "risky",    "risk_score": 85, "account_age_days": 6,   "is_new": 1, "note": "⚠ Flagged: multiple STRs",   "sort_order": 9},
    {"beneficiary_id": "BENF-SCAM-088",   "name": "Lucky Prize Payout",  "bank_label": "Unverified bank",    "tier": "risky",    "risk_score": 78, "account_age_days": 2,   "is_new": 1, "note": "⚠ Flagged: APP fraud payout", "sort_order": 10},
]
db.seed_beneficiaries(DEFAULT_BENEFICIARIES)

# Global model references
lgb_model = iso_model = scaler = feature_cols = anomaly_features = None
optimal_threshold = 0.5
shap_explainer = None
feat_importance = None

@app.on_event("startup")
async def load_models():
    global lgb_model, iso_model, scaler, feature_cols, anomaly_features
    global optimal_threshold, shap_explainer, feat_importance
    try:
        lgb_model = joblib.load(MODEL_DIR / "lgb_model.pkl")
        iso_model = joblib.load(MODEL_DIR / "iso_model.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
        anomaly_features = joblib.load(MODEL_DIR / "anomaly_features.pkl")
        optimal_threshold = joblib.load(MODEL_DIR / "optimal_threshold.pkl")
        feat_importance = joblib.load(MODEL_DIR / "feature_importance.pkl")
        try:
            shap_explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")
            print("[OK] SHAP explainer loaded")
        except: pass
        print(f"[OK] Models loaded ({len(feature_cols)} features, threshold={optimal_threshold:.3f})")
        # Restore live feed + running stats from the database.
        try:
            for r in reversed(db.recent_transactions(100)):  # oldest→newest so newest ends leftmost
                txn_feed.appendleft(r)
            session_stats.update(db.decision_counts())
            print(f"[OK] Restored {len(txn_feed)} transactions from DB "
                  f"({len(trust_engine.customers)} customers, {len(reputation_store.entities)} ledger entities)")
        except Exception as e:
            print("[WARN] DB restore:", e)
    except FileNotFoundError:
        print("[WARN] Models not found - run: python models/train_pipeline.py")
        # Try legacy models
        try:
            lgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")
            scaler = joblib.load(MODEL_DIR / "scaler.pkl")
            feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
            print("[OK] Legacy models loaded (limited features)")
        except:
            print("[WARN] No models available - using rule-based mode")

session_stats = {"total": 0, "block": 0, "verify": 0, "approve": 0, "start": time.time()}
blockchain_log = deque(maxlen=100)
governance_log = deque(maxlen=200)
txn_feed = deque(maxlen=100)   # live feed of recent scored txns (for the dashboard)

# ── Schemas ──
class Transaction(BaseModel):
    amount: float = Field(..., gt=0, example=15000.0)
    hour: int = Field(..., ge=0, le=23, example=14)
    customer_id: Optional[str] = Field("anonymous", example="CUST-001")
    # Reputation-ledger identifiers (looked up in the reputation store)
    beneficiary_id: Optional[str] = Field(None, example="BENF-9021")
    device_hash: Optional[str] = Field(None, example="DEV-3f9a")
    # Origin channel (e.g. "IOB Pay", "api") — shown in the dashboard live feed
    channel: Optional[str] = Field("api", example="IOB Pay")
    beneficiary_name: Optional[str] = Field(None, example="Priya Sharma")
    # Behavioural
    avg_txn_amount: Optional[float] = Field(5000, example=5000)
    txn_frequency: Optional[float] = Field(1.0, example=1.5)
    txn_count_1h: Optional[int] = Field(1, example=1)
    txn_count_24h: Optional[int] = Field(2, example=3)
    # Device
    new_device: bool = Field(False)
    rooted_device: bool = Field(False)
    emulator_detected: bool = Field(False)
    device_trust_score: Optional[float] = Field(80, ge=0, le=100)
    # Geo
    geo_distance_km: Optional[float] = Field(5, example=10)
    vpn_detected: bool = Field(False)
    is_international: bool = Field(False)
    # Session
    session_duration_sec: Optional[float] = Field(45, example=45)
    paste_detected: bool = Field(False)
    failed_otp_count: Optional[int] = Field(0, ge=0)
    beneficiary_added_recently: bool = Field(False)
    typing_speed_anomaly: Optional[float] = Field(0.1, ge=0, le=1)
    # Relationship
    shared_device_accounts: Optional[int] = Field(1)
    beneficiary_risk_score: Optional[float] = Field(10, ge=0, le=100)
    account_age_days: Optional[int] = Field(365)
    # Location context (IOB Pay map widget) — the customer's usual/frequent
    # location vs. where this specific payment is being made from. Kept
    # alongside geo_distance_km/is_international purely so the raw picture
    # (pins + labels) survives into the stored transaction for the dashboard.
    home_label: Optional[str] = None
    home_lat: Optional[float] = None
    home_lon: Optional[float] = None
    payment_label: Optional[str] = None
    payment_lat: Optional[float] = None
    payment_lon: Optional[float] = None
    # PCA (optional, for full ML scoring)
    v1: Optional[float]=None; v2: Optional[float]=None; v3: Optional[float]=None
    v4: Optional[float]=None; v5: Optional[float]=None; v6: Optional[float]=None
    v7: Optional[float]=None; v8: Optional[float]=None; v9: Optional[float]=None
    v10: Optional[float]=None; v11: Optional[float]=None; v12: Optional[float]=None
    v13: Optional[float]=None; v14: Optional[float]=None; v15: Optional[float]=None
    v16: Optional[float]=None; v17: Optional[float]=None; v18: Optional[float]=None
    v19: Optional[float]=None; v20: Optional[float]=None; v21: Optional[float]=None
    v22: Optional[float]=None; v23: Optional[float]=None; v24: Optional[float]=None
    v25: Optional[float]=None; v26: Optional[float]=None; v27: Optional[float]=None
    v28: Optional[float]=None

class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: int
    tti: float                      # Transaction Trust Index (PRD §12)
    tti_breakdown: List[dict]       # the 6 named factors + weights + contributions
    decision: str
    fraud_probability: float
    anomaly_score: float
    trust_score: float
    trust_delta: float
    decision_confidence: float
    reasons: List[str]
    shap_explanations: List[dict]
    trust_factors: List[dict]
    response_time_ms: float
    blockchain_hash: Optional[str]
    governance_report: dict
    timestamp: str

# ── IOB Pay account schemas (simulation only — not production auth) ──
class SignupRequest(BaseModel):
    full_name: str = Field(..., min_length=1, example="Priya Sharma")
    email: str = Field(..., example="priya@example.com")
    phone: Optional[str] = None
    password: str = Field(..., min_length=4)
    home_label: str = Field(..., example="Bengaluru")
    home_lat: float
    home_lon: float
    home_country: Optional[str] = "IN"

class LoginRequest(BaseModel):
    email: str
    password: str

# ── Feature Builder ──
def build_feature_vector(tx: Transaction) -> np.ndarray:
    """Build feature vector matching training pipeline order."""
    v_vals = [getattr(tx, f"v{i}", None) or 0.0 for i in range(1, 29)]
    log_amount = math.log1p(tx.amount)
    amount_zscore = (tx.amount - 88.35) / 250.12
    avg = tx.avg_txn_amount or 5000
    amount_dev = tx.amount / max(avg, 1)
    impossible = 1 if (tx.geo_distance_km or 0) > 500 and (tx.txn_count_1h or 1) > 1 else 0
    geo_risk = 70 if tx.vpn_detected else (50 if tx.is_international else 15)
    session_risk = (
        (1 if tx.paste_detected else 0) * 20 +
        (tx.failed_otp_count or 0) * 15 +
        (1 if tx.beneficiary_added_recently else 0) * 20 +
        (tx.typing_speed_anomaly or 0.1) * 30 +
        (15 if (tx.session_duration_sec or 45) < 10 else 0)
    )
    active_hour = 0.8 if 8 <= tx.hour <= 22 else 0.3
    device_trust = tx.device_trust_score or 80
    risk_composite = (
        (100 - device_trust) * 0.2 + geo_risk * 0.2 +
        min(session_risk, 100) * 0.3 + (tx.beneficiary_risk_score or 10) * 0.15 +
        min((tx.txn_frequency or 1) / 30, 1) * 100 * 0.15
    )
    features = {
        **{f"V{i}": v_vals[i-1] for i in range(1, 29)},
        "Amount": tx.amount, "hour": tx.hour, "day_of_week": 3,
        "active_hour_score": active_hour,
        "avg_txn_amount": avg, "amount_deviation": amount_dev,
        "txn_frequency": tx.txn_frequency or 1, "txn_count_1h": tx.txn_count_1h or 1,
        "txn_count_24h": tx.txn_count_24h or 2,
        "new_device_flag": int(tx.new_device), "rooted_device": int(tx.rooted_device),
        "emulator_detected": int(tx.emulator_detected), "device_trust_score": device_trust,
        "geo_distance_km": tx.geo_distance_km or 5, "impossible_travel": impossible,
        "vpn_detected": int(tx.vpn_detected), "is_international": int(tx.is_international),
        "geo_risk_score": geo_risk,
        "session_duration_sec": tx.session_duration_sec or 45,
        "paste_detected": int(tx.paste_detected), "failed_otp_count": tx.failed_otp_count or 0,
        "beneficiary_added_recently": int(tx.beneficiary_added_recently),
        "typing_speed_anomaly": tx.typing_speed_anomaly or 0.1,
        "session_risk_score": min(session_risk, 100),
        "shared_device_accounts": tx.shared_device_accounts or 1,
        "beneficiary_risk_score": tx.beneficiary_risk_score or 10,
        "account_age_days": tx.account_age_days or 365,
        "log_amount": log_amount, "amount_zscore": amount_zscore,
        "risk_composite": risk_composite,
    }
    vec = [features.get(col, 0.0) for col in feature_cols]
    return np.array(vec).reshape(1, -1)

# ── SHAP Explanation ──
def get_shap_explanation(X_scaled, top_n=5):
    if shap_explainer is None:
        return get_importance_explanation(X_scaled, top_n)
    try:
        import shap
        sv = shap_explainer.shap_values(X_scaled)
        if isinstance(sv, list): sv = sv[1]
        sv = sv[0]
        indices = np.argsort(np.abs(sv))[::-1][:top_n]
        return [{"feature": feature_cols[i], "shap_value": round(float(sv[i]), 4),
                 "impact": "increases_risk" if sv[i] > 0 else "decreases_risk"}
                for i in indices]
    except:
        return get_importance_explanation(X_scaled, top_n)

def get_importance_explanation(X_scaled, top_n=5):
    if feat_importance is None: return []
    top = feat_importance.head(top_n)
    return [{"feature": row["feature"], "shap_value": round(float(X_scaled[0][feature_cols.index(row["feature"])]), 4),
             "impact": "high_importance"} for _, row in top.iterrows() if row["feature"] in feature_cols]

# ── Scoring ──
def ml_score(tx: Transaction):
    X_raw = build_feature_vector(tx)
    X_scaled = scaler.transform(X_raw)
    fraud_prob = float(lgb_model.predict_proba(X_scaled)[0][1])
    # Anomaly score
    anomaly = 0.5
    if iso_model and anomaly_features:
        try:
            anom_idx = [feature_cols.index(f) for f in anomaly_features if f in feature_cols]
            X_anom = X_scaled[:, anom_idx]
            raw_iso = float(iso_model.decision_function(X_anom)[0])
            anomaly = float(np.clip((-raw_iso + 0.5) * 2, 0, 1))
        except: pass
    shap_expl = get_shap_explanation(X_scaled)
    reasons = []
    if fraud_prob > 0.7: reasons.append(f"LightGBM: high fraud probability ({fraud_prob:.1%})")
    elif fraud_prob > 0.4: reasons.append(f"LightGBM: elevated fraud probability ({fraud_prob:.1%})")
    else: reasons.append(f"LightGBM: low fraud probability ({fraud_prob:.1%})")
    if anomaly > 0.6: reasons.append("Isolation Forest: anomalous pattern detected")
    if tx.new_device: reasons.append("New device — fingerprint mismatch")
    if tx.vpn_detected: reasons.append("VPN/proxy detected — masked IP")
    if tx.emulator_detected: reasons.append("Emulator detected — fake environment")
    if tx.rooted_device: reasons.append("Rooted/jailbroken device")
    if tx.paste_detected: reasons.append("Account details pasted (potential phishing)")
    if (tx.failed_otp_count or 0) >= 2: reasons.append(f"Failed OTP attempts: {tx.failed_otp_count}")
    if tx.beneficiary_added_recently: reasons.append("New beneficiary added before large transfer")
    if (tx.geo_distance_km or 0) > 500: reasons.append(f"Impossible travel: {tx.geo_distance_km:.0f} km from last txn")
    if tx.is_international: reasons.append("International transaction")
    avg = tx.avg_txn_amount or 5000
    if tx.amount / avg > 5: reasons.append(f"Amount {tx.amount/avg:.1f}x above user baseline")
    return fraud_prob, anomaly, reasons, shap_expl

def rule_based_score(tx: Transaction):
    score, reasons = 0, []
    avg = tx.avg_txn_amount or 5000
    ratio = tx.amount / avg
    if ratio > 10: score += 40; reasons.append(f"Amount {ratio:.1f}x above average")
    elif ratio > 5: score += 25; reasons.append(f"Amount {ratio:.1f}x above average")
    elif ratio > 2: score += 12; reasons.append(f"Amount {ratio:.1f}x above average")
    if tx.hour < 5: score += 20; reasons.append("Late-night transaction")
    if tx.new_device: score += 15; reasons.append("New device detected")
    if tx.vpn_detected: score += 10; reasons.append("VPN detected")
    if tx.emulator_detected: score += 12; reasons.append("Emulator detected")
    if tx.rooted_device: score += 8; reasons.append("Rooted device")
    if tx.paste_detected: score += 10; reasons.append("Paste detected")
    if (tx.failed_otp_count or 0) >= 2: score += 15; reasons.append(f"Failed OTP x{tx.failed_otp_count}")
    if tx.beneficiary_added_recently: score += 12; reasons.append("New beneficiary")
    if tx.is_international: score += 8; reasons.append("International txn")
    if (tx.geo_distance_km or 0) > 500: score += 15; reasons.append("Impossible travel")
    return min(score, 100) / 100, min(score, 100) / 100, reasons, []

# ── Transaction Trust Index (PRD §12) ──
# The TTI is a single 0-100 index that combines the six signals named in the
# PRD. Each factor below is expressed as a RISK contribution (0 = fully trusted,
# 100 = maximal risk), so a HIGH TTI means a HIGH-risk transaction — matching the
# PRD decision rule: TTI Low → Approve, Medium → Step-up MFA, High → Block.
TTI_WEIGHTS = {
    "Transaction Risk (LightGBM+IsoForest)": 0.35,
    "Device Trust":                          0.15,
    "Historical Behaviour":                  0.15,
    "Beneficiary Reputation":                0.15,
    "Authentication History":                0.10,
    "Blockchain Threat Intelligence":        0.10,
}

def blockchain_intel_lookup(tx: Transaction):
    """Blockchain / threat-intel factor for the TTI.

    Reads the permissioned reputation ledger for the beneficiary and device and
    returns (risk, hits) where risk is the WORST reputation found across the two
    entities and hits lists the ledger records that matched. Phase 3 wires the
    write side (report_fraud) so confirmed fraud raises these scores over time.
    """
    risk = 8.0   # baseline "no adverse intelligence found"
    hits = []
    for eid, kind in [(tx.beneficiary_id, "beneficiary"), (tx.device_hash, "device")]:
        if not eid:
            continue
        rec = reputation_store.lookup(eid, kind)
        if rec["known"] and rec["risk"] > 0:
            risk = max(risk, rec["risk"])
            hits.append(rec)
    return risk, hits

def compute_tti(tx: Transaction, fraud_prob: float, anomaly: float, trust_info: dict, reasons: list = None):
    """Compute the Transaction Trust Index and its per-factor breakdown.

    Returns (tti_value, breakdown) where breakdown is a list of
    {factor, score, weight, contribution} dicts — one per PRD factor.
    """
    # 1. Transaction risk — the AI Risk Engine (LightGBM + Isolation Forest)
    transaction_risk = min((fraud_prob * 0.7 + anomaly * 0.3) * 100, 100)

    # 2. Beneficiary reputation — real lookup in the reputation ledger.
    #    A known-bad beneficiary uses its ledger risk directly; an unknown /
    #    first-time beneficiary has no track record and carries a first-time
    #    premium (PRD scope: "High-Risk First-Time Beneficiary Transfers").
    benf_rep = reputation_store.lookup(tx.beneficiary_id, "beneficiary")
    if benf_rep["known"] and benf_rep["risk"] > 0:
        beneficiary_risk = benf_rep["risk"]
    else:
        beneficiary_risk = tx.beneficiary_risk_score or 10
        first_time = benf_rep["first_time"] if tx.beneficiary_id else tx.beneficiary_added_recently
        if first_time or tx.beneficiary_added_recently:
            beneficiary_risk = min(beneficiary_risk + 25, 100)

    # 3. Authentication history — failed OTPs and pasted credentials.
    auth_risk = min((tx.failed_otp_count or 0) * 22 + (15 if tx.paste_detected else 0), 100)

    # 4. Device trust — invert device_trust_score, penalise compromised devices.
    device_risk = 100 - (tx.device_trust_score or 80)
    if tx.new_device:        device_risk += 15
    if tx.rooted_device:     device_risk += 15
    if tx.emulator_detected: device_risk += 20
    device_risk = min(device_risk, 100)

    # 5. Historical behaviour — the Dynamic Trust Engine's continuous customer
    #    trust, plus amount deviation and velocity against the user's baseline.
    hist_risk = 100 - trust_info["trust_score"]
    amount_ratio = tx.amount / max(tx.avg_txn_amount or 5000, 1)
    if amount_ratio > 5:            hist_risk += 15
    if (tx.txn_count_1h or 1) > 3:  hist_risk += 10
    hist_risk = min(hist_risk, 100)

    # 6. Blockchain threat intelligence — reputation from the permissioned ledger.
    blockchain_risk, intel_hits = blockchain_intel_lookup(tx)

    factors = {
        "Transaction Risk (LightGBM+IsoForest)": transaction_risk,
        "Device Trust":                          device_risk,
        "Historical Behaviour":                  hist_risk,
        "Beneficiary Reputation":                beneficiary_risk,
        "Authentication History":                auth_risk,
        "Blockchain Threat Intelligence":        blockchain_risk,
    }

    # Per-factor notes + ledger-driven reasons (surface why reputation moved).
    notes = {}
    if benf_rep["known"] and benf_rep["risk"] > 0:
        notes["Beneficiary Reputation"] = f"Ledger flag: {benf_rep.get('note','known-bad')} (risk {benf_rep['risk']:.0f})"
        if reasons is not None:
            reasons.append(f"Beneficiary {tx.beneficiary_id} on fraud ledger — {benf_rep.get('note','known-bad')}")
    elif tx.beneficiary_id and benf_rep["first_time"]:
        notes["Beneficiary Reputation"] = "First-time beneficiary — no track record"
        if reasons is not None:
            reasons.append("First-time beneficiary — no reputation history")
    for hit in intel_hits:
        notes["Blockchain Threat Intelligence"] = f"Ledger hit: {hit['entity_id']} ({hit.get('note','flagged')})"
        if reasons is not None:
            reasons.append(f"Blockchain intel: {hit['entity_id']} flagged ({hit['fraud_reports']} prior reports)")

    tti = sum(score * TTI_WEIGHTS[f] for f, score in factors.items())
    breakdown = [
        {"factor": f, "score": round(score, 1), "weight": TTI_WEIGHTS[f],
         "contribution": round(score * TTI_WEIGHTS[f], 1), "note": notes.get(f, "")}
        for f, score in factors.items()
    ]
    return min(tti, 100.0), breakdown

def decide(score: int) -> str:
    if score < 35: return "APPROVE"
    if score < 70: return "STEP_UP_MFA"
    return "BLOCK"

# ── Blockchain policy override (PRD §10/§12) ──
# A confirmed-fraud entity on the permissioned ledger overrides the weighted TTI:
# the shared intelligence is high-confidence, so any transfer touching it is
# escalated regardless of how clean the rest of the transaction looks.
LEDGER_BLOCK_THRESHOLD = 70.0   # ledger risk >= this forces BLOCK
LEDGER_MFA_THRESHOLD   = 40.0   # ledger risk >= this forces at least STEP_UP_MFA
# Severity written back to the ledger when a transfer is confirmed fraud/blocked.
FRAUD_WRITEBACK_SEVERITY = {"beneficiary": 75.0, "device": 45.0}

DECISION_RANK = {"APPROVE": 0, "STEP_UP_MFA": 1, "BLOCK": 2}

def apply_ledger_override(decision: str, intel_risk: float, intel_hits: list, reasons: list):
    """Escalate the decision when a confirmed-fraud ledger entity is involved."""
    if not intel_hits:
        return decision
    if intel_risk >= LEDGER_BLOCK_THRESHOLD:
        forced = "BLOCK"
    elif intel_risk >= LEDGER_MFA_THRESHOLD:
        forced = "STEP_UP_MFA"
    else:
        return decision
    if DECISION_RANK[forced] > DECISION_RANK[decision]:
        ids = ", ".join(h["entity_id"] for h in intel_hits)
        reasons.append(f"Policy override → {forced}: blockchain-confirmed entity ({ids}, risk {intel_risk:.0f})")
        return forced
    return decision

def blockchain_store(tx_id, score, amount, reasons):
    data = f"{tx_id}:{score}:{amount}:{datetime.utcnow().isoformat()}"
    tx_hash = "0x" + hashlib.sha256(data.encode()).hexdigest()[:16].upper()
    blockchain_log.appendleft({"hash": tx_hash, "tx_id": tx_id, "risk_score": score,
        "amount": amount, "reasons": reasons[:3], "timestamp": datetime.utcnow().isoformat(), "status": "COMMITTED"})
    return tx_hash

def generate_str_report(tx_id, tx, score, decision, reasons, trust_info):
    report = {
        "str_id": f"STR-{tx_id}", "type": "Suspicious Transaction Report",
        "transaction_id": tx_id, "amount": tx.amount, "customer_id": tx.customer_id,
        "risk_score": score, "decision": decision,
        "risk_indicators": reasons[:5], "trust_score": trust_info["trust_score"],
        "trust_factors": trust_info["trust_factors"][:5],
        "generated_at": datetime.utcnow().isoformat(),
        "regulatory_basis": "RBI Master Direction on Fraud Risk Management",
        "action_required": "Review within 24 hours" if decision == "STEP_UP_MFA" else "Immediate investigation"
    }
    governance_log.appendleft(report)
    return report

# ── Endpoints ──
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": lgb_model is not None,
            "mode": "LightGBM+IsoForest+SHAP" if lgb_model else "rule-based",
            "features": len(feature_cols) if feature_cols else 0,
            "trust_engine": "active", "customers_tracked": len(trust_engine.customers),
            "uptime_seconds": round(time.time() - session_stats["start"])}

@app.get("/stats")
def stats():
    t = session_stats["total"]
    return {"total": t, "blocked": session_stats["block"], "mfa": session_stats["verify"],
            "approved": session_stats["approve"],
            "fraud_rate": round(session_stats["block"] / max(t, 1) * 100, 2),
            "customers_tracked": len(trust_engine.customers)}

@app.get("/transactions")
def list_transactions(limit: int = 300):
    """List recent scored transactions (summaries) from the database."""
    try:
        return {"transactions": db.list_transactions(limit)}
    except Exception as e:
        return {"transactions": [], "error": str(e)}

@app.get("/transaction/{transaction_id}")
def get_transaction(transaction_id: str):
    """Full stored detail for one transaction (breakdown, SHAP, reasons, STR)."""
    rec = db.get_transaction(transaction_id)
    if not rec:
        raise HTTPException(404, "Transaction not found")
    return rec

@app.get("/feed")
def get_feed(limit: int = 25):
    """Live feed of recently scored transactions (newest first)."""
    t = session_stats["total"]
    return {
        "feed": list(txn_feed)[:limit],
        "count": len(txn_feed),
        "stats": {"total": t, "blocked": session_stats["block"],
                  "mfa": session_stats["verify"], "approved": session_stats["approve"]},
    }

@app.post("/score", response_model=ScoreResponse)
def score_transaction(tx: Transaction):
    start = time.perf_counter()
    tx_id = str(uuid.uuid4())[:8].upper()
    if lgb_model is not None and feature_cols:
        fraud_prob, anomaly, reasons, shap_expl = ml_score(tx)
    else:
        fraud_prob, anomaly, reasons, shap_expl = rule_based_score(tx)
    # Trust Engine
    impossible = (tx.geo_distance_km or 0) > 500 and (tx.txn_count_1h or 1) > 1
    geo_risk = 70 if tx.vpn_detected else 15
    session_risk = min(((1 if tx.paste_detected else 0)*20 + (tx.failed_otp_count or 0)*15 +
        (1 if tx.beneficiary_added_recently else 0)*20 + (tx.typing_speed_anomaly or 0.1)*30), 100)
    trust_info = trust_engine.compute_trust_adjustment(
        customer_id=tx.customer_id or "anonymous", ml_fraud_score=fraud_prob,
        anomaly_flag=anomaly > 0.6, session_risk=session_risk,
        device_trust=tx.device_trust_score or 80, geo_risk=geo_risk,
        vpn_detected=tx.vpn_detected, new_device=tx.new_device,
        impossible_travel=impossible, txn_amount=tx.amount,
        avg_amount=tx.avg_txn_amount or 5000, failed_otp=tx.failed_otp_count or 0,
        paste_detected=tx.paste_detected, beneficiary_new=tx.beneficiary_added_recently)
    # Transaction Trust Index (PRD §12): explicit 6-factor blend replaces the
    # old ad-hoc score × trust-multiplier — historical behaviour is now a factor.
    tti_value, tti_breakdown = compute_tti(tx, fraud_prob, anomaly, trust_info, reasons)
    risk_score = int(round(tti_value))
    decision = decide(risk_score)
    # ── Blockchain policy override ──
    # Read the ledger BEFORE any write-back so a txn is never escalated by its
    # own fingerprint — only by intelligence committed on prior transactions.
    intel_risk, intel_hits = blockchain_intel_lookup(tx)
    decision = apply_ledger_override(decision, intel_risk, intel_hits, reasons)
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    session_stats["total"] += 1
    key = "block" if decision == "BLOCK" else ("verify" if decision == "STEP_UP_MFA" else "approve")
    session_stats[key] += 1
    chain_hash = None
    if decision in ("BLOCK", "STEP_UP_MFA"):
        chain_hash = blockchain_store(tx_id, risk_score, tx.amount, reasons)
    # ── Feedback loop (PRD §7 step 8): commit fraud fingerprints back to the ──
    # ledger so the NEXT transfer to the same beneficiary/device inherits the risk.
    if decision == "BLOCK":
        committed = []
        for eid, kind in [(tx.beneficiary_id, "beneficiary"), (tx.device_hash, "device")]:
            rec = reputation_store.report_fraud(eid, kind, FRAUD_WRITEBACK_SEVERITY[kind])
            if rec:
                committed.append(f"{rec['entity_id']}→{rec['risk']:.0f}")
        if committed:
            reasons.append(f"Fraud fingerprint committed to ledger: {', '.join(committed)}")
    gov_report = {}
    if decision == "BLOCK" or risk_score > 60:
        gov_report = generate_str_report(tx_id, tx, risk_score, decision, reasons, trust_info)
    trust_factors_out = [{"factor": f, "impact": v} for f, v in trust_info["trust_factors"]]
    # Record in the live feed so any client (dashboard) can display it in real time.
    feed_entry = {
        "transaction_id": tx_id, "timestamp": datetime.utcnow().isoformat(),
        "channel": tx.channel or "api", "amount": tx.amount,
        "customer_id": tx.customer_id, "beneficiary_id": tx.beneficiary_id,
        "beneficiary_name": tx.beneficiary_name, "tti": round(tti_value, 1),
        "decision": decision, "fraud_probability": round(fraud_prob, 4),
        "trust_score": trust_info["trust_score"],
        "top_reason": reasons[0] if reasons else "",
    }
    txn_feed.appendleft(feed_entry)
    resp = ScoreResponse(
        transaction_id=tx_id, risk_score=risk_score,
        tti=round(tti_value, 1), tti_breakdown=tti_breakdown, decision=decision,
        fraud_probability=round(fraud_prob, 4), anomaly_score=round(anomaly, 4),
        trust_score=trust_info["trust_score"], trust_delta=trust_info["trust_delta"],
        decision_confidence=trust_info["decision_confidence"],
        reasons=reasons, shap_explanations=shap_expl, trust_factors=trust_factors_out,
        response_time_ms=elapsed, blockchain_hash=chain_hash,
        governance_report=gov_report, timestamp=datetime.utcnow().isoformat())
    # Persist full detail so the dashboard can re-open any past transaction.
    try:
        detail = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
        detail["amount"] = tx.amount
        detail["customer_id"] = tx.customer_id
        detail["beneficiary_id"] = tx.beneficiary_id
        detail["beneficiary_name"] = tx.beneficiary_name
        detail["channel"] = tx.channel or "api"
        # Full raw input (account age, session, device, location, ...) so the
        # dashboard can show exactly what the engine saw, not just the verdict.
        detail["raw"] = tx.model_dump() if hasattr(tx, "model_dump") else tx.dict()
        db.insert_transaction(feed_entry, detail=detail)
    except Exception as e:
        print("[WARN] txn persist:", e)
    return resp

@app.get("/blockchain/log")
def get_blockchain_log():
    return {"entries": list(blockchain_log)[:20], "count": len(blockchain_log)}

@app.get("/governance/reports")
def get_governance_reports():
    return {"reports": list(governance_log)[:20], "count": len(governance_log)}

@app.get("/reputation")
def get_reputation_ledger():
    """Snapshot of the beneficiary/device reputation ledger."""
    return reputation_store.summary()

@app.get("/reputation/{entity_id}")
def get_reputation(entity_id: str, kind: str = "beneficiary"):
    return reputation_store.lookup(entity_id, kind)

@app.get("/trust/{customer_id}")
def get_trust(customer_id: str):
    s = trust_engine.get_customer_summary(customer_id)
    if not s: raise HTTPException(404, "Customer not found")
    return s

@app.post("/score/batch")
def score_batch(transactions: List[Transaction]):
    if len(transactions) > 100: raise HTTPException(400, "Max 100 per batch")
    return {"results": [score_transaction(tx) for tx in transactions]}

# ── IOB Pay accounts (simulation) ──
# A real bank never stores or checks passwords like this. This exists only so
# the IOB Pay demo can require "log in before you can pay" — i.e. every
# transaction is tied to one consistently-identified customer_id instead of a
# free-text field — which is what makes the customer's history (account age,
# trust score, home location) mean anything to the fraud engine.
def _hash_password(password: str) -> str:
    return hashlib.sha256(f"paymentguardian-sim::{password}".encode()).hexdigest()

def _user_public(u: dict) -> dict:
    return {"user_id": u["user_id"], "full_name": u["full_name"], "email": u["email"],
            "home_label": u["home_label"], "home_lat": u["home_lat"], "home_lon": u["home_lon"],
            "home_country": u["home_country"], "balance": u["balance"]}

@app.post("/auth/signup")
def signup(req: SignupRequest):
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "Enter a valid email address")
    if db.get_user_by_email(email):
        raise HTTPException(409, "An account with this email already exists — please log in instead")
    user_id = "IOBPAY-" + hashlib.sha1(email.encode()).hexdigest()[:8].upper()
    if db.get_user(user_id):
        raise HTTPException(409, "An account already exists for this email")
    user = {"user_id": user_id, "full_name": req.full_name.strip(), "email": email,
            "phone": req.phone, "password_hash": _hash_password(req.password),
            "home_label": req.home_label, "home_lat": req.home_lat, "home_lon": req.home_lon,
            "home_country": req.home_country or "IN", "balance": DEFAULT_STARTING_BALANCE,
            "created_at": datetime.utcnow().isoformat()}
    db.create_user(user)
    return _user_public(user)

@app.post("/auth/login")
def login(req: LoginRequest):
    user = db.get_user_by_email(req.email.strip().lower())
    if not user or user["password_hash"] != _hash_password(req.password):
        raise HTTPException(401, "Incorrect email or password")
    return _user_public(user)

@app.get("/beneficiaries")
def get_beneficiaries():
    """The 10 'pay to' accounts shown in IOB Pay, pre-classified safe / moderate / risky."""
    return {"beneficiaries": db.list_beneficiaries()}

# ── Wallet balance ──
# IOB Pay's account balance was static before; a completed payment now
# actually debits it, and it's read back on login/refresh so it survives
# reloads and stays in sync with what the DB has.
class DebitRequest(BaseModel):
    user_id: str
    amount: float = Field(..., gt=0)

@app.get("/wallet/balance/{user_id}")
def get_wallet_balance(user_id: str):
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(404, "Account not found")
    return {"user_id": user_id, "balance": user["balance"]}

@app.post("/wallet/debit")
def debit_wallet(req: DebitRequest):
    """Called by IOB Pay right after a payment is APPROVED (or MFA-verified) —
    never for a BLOCK, and never as part of /score itself, since /score also
    scores synthetic/simulated traffic that isn't tied to a real balance."""
    user = db.get_user(req.user_id)
    if not user:
        raise HTTPException(404, "Account not found")
    new_balance = user["balance"] - req.amount
    if new_balance < 0:
        raise HTTPException(402, "Insufficient balance")
    db.update_balance(req.user_id, new_balance)
    return {"user_id": req.user_id, "balance": new_balance}

# ── Serve the IOB Pay simulator (React single-page app) at /pay ──
# Mounted last so it never shadows the API routes above.
IOBPAY_DIR = Path(__file__).parent.parent / "iobpay"
if IOBPAY_DIR.exists():
    app.mount("/pay", StaticFiles(directory=str(IOBPAY_DIR), html=True), name="pay")
    print("[OK] PaymentGuardian Pay app served at /pay")

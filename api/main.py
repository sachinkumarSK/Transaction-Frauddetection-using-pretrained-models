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
from pydantic import BaseModel, Field

# Add parent to path for trust engine import
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
from trust_engine import DynamicTrustEngine

app = FastAPI(title="Continuous Trust Intelligence API", version="2.0.0",
              description="Real-time fraud prevention with Dynamic Trust Scoring + LightGBM + SHAP")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = Path(__file__).parent.parent / "models"
trust_engine = DynamicTrustEngine()

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

# ── Schemas ──
class Transaction(BaseModel):
    amount: float = Field(..., gt=0, example=15000.0)
    hour: int = Field(..., ge=0, le=23, example=14)
    customer_id: Optional[str] = Field("anonymous", example="CUST-001")
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

def decide(score: int) -> str:
    if score < 35: return "APPROVE"
    if score < 70: return "STEP_UP_MFA"
    return "BLOCK"

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

@app.post("/score", response_model=ScoreResponse)
def score_transaction(tx: Transaction):
    start = time.perf_counter()
    tx_id = str(uuid.uuid4())[:8].upper()
    if lgb_model is not None and feature_cols:
        fraud_prob, anomaly, reasons, shap_expl = ml_score(tx)
        combined = fraud_prob * 0.65 + anomaly * 0.35
        risk_score = int(min(combined * 100, 100))
    else:
        fraud_prob, anomaly, reasons, shap_expl = rule_based_score(tx)
        risk_score = int(fraud_prob * 100)
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
    # Apply trust multiplier to risk score
    risk_score = int(min(risk_score * trust_info["risk_multiplier"], 100))
    decision = decide(risk_score)
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    session_stats["total"] += 1
    key = "block" if decision == "BLOCK" else ("verify" if decision == "STEP_UP_MFA" else "approve")
    session_stats[key] += 1
    chain_hash = None
    if decision in ("BLOCK", "STEP_UP_MFA"):
        chain_hash = blockchain_store(tx_id, risk_score, tx.amount, reasons)
    gov_report = {}
    if decision == "BLOCK" or risk_score > 60:
        gov_report = generate_str_report(tx_id, tx, risk_score, decision, reasons, trust_info)
    trust_factors_out = [{"factor": f, "impact": v} for f, v in trust_info["trust_factors"]]
    return ScoreResponse(
        transaction_id=tx_id, risk_score=risk_score, decision=decision,
        fraud_probability=round(fraud_prob, 4), anomaly_score=round(anomaly, 4),
        trust_score=trust_info["trust_score"], trust_delta=trust_info["trust_delta"],
        decision_confidence=trust_info["decision_confidence"],
        reasons=reasons, shap_explanations=shap_expl, trust_factors=trust_factors_out,
        response_time_ms=elapsed, blockchain_hash=chain_hash,
        governance_report=gov_report, timestamp=datetime.utcnow().isoformat())

@app.get("/blockchain/log")
def get_blockchain_log():
    return {"entries": list(blockchain_log)[:20], "count": len(blockchain_log)}

@app.get("/governance/reports")
def get_governance_reports():
    return {"reports": list(governance_log)[:20], "count": len(governance_log)}

@app.get("/trust/{customer_id}")
def get_trust(customer_id: str):
    s = trust_engine.get_customer_summary(customer_id)
    if not s: raise HTTPException(404, "Customer not found")
    return s

@app.post("/score/batch")
def score_batch(transactions: List[Transaction]):
    if len(transactions) > 100: raise HTTPException(400, "Max 100 per batch")
    return {"results": [score_transaction(tx) for tx in transactions]}

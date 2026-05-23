# 🛡️ Continuous Trust Intelligence Framework
## Real-Time Fraud Prevention in Digital Payment Systems

---

## Architecture

```
Digital Channels (UPI / Card / Wallet)
            ↓
    FastAPI Ingestion Layer
            ↓
    Feature Engineering (50+ features)
            ↓
    ┌───────────────────────────┐
    │ LightGBM (fraud scoring)  │──→ Fraud Probability
    │ Isolation Forest (anomaly)│──→ Anomaly Flag
    │ SHAP (explainability)     │──→ Why blocked?
    └───────────────────────────┘
            ↓
    Dynamic Trust Engine
    (continuous customer trust scoring)
            ↓
    Decision Router
    ┌────────┬────────────┬────────┐
    │APPROVE │ STEP_UP_MFA│ BLOCK  │
    └────────┴────────────┴────────┘
            ↓
    Governance Layer (STR Reports + Audit)
    Blockchain Layer (Fraud Fingerprints)
```

---

## Quick Start (Step by Step)

### 1. Install dependencies
```bash
cd fraud-demo
pip install -r requirements.txt
```

### 2. Generate enriched dataset (synthetic + Kaggle)
```bash
python data/generate_synthetic.py
```
This takes the Kaggle `creditcard.csv` and adds 25+ behavioral features.

### 3. Train all models (LightGBM + IsoForest + SHAP)
```bash
python models/train_pipeline.py
```
Takes ~5-10 minutes. Includes GridSearchCV fine-tuning.

### 4. Start the API (Terminal 1)
```bash
cd api
uvicorn main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### 5. Start the Dashboard (Terminal 2)
```bash
streamlit run dashboard/app.py
```
Dashboard: http://localhost:8501

---

## Project Structure

```
fraud-demo/
├── data/
│   ├── creditcard.csv               ← Kaggle dataset (download)
│   ├── generate_synthetic.py        ← Enriches with 25+ features
│   └── enriched_fraud_data.csv      ← Generated output
├── models/
│   ├── train_pipeline.py            ← LightGBM + IsoForest + SHAP
│   ├── trust_engine.py              ← Dynamic Trust Engine
│   ├── lgb_model.pkl                ← Trained LightGBM
│   ├── iso_model.pkl                ← Trained Isolation Forest
│   ├── shap_explainer.pkl           ← SHAP TreeExplainer
│   ├── scaler.pkl                   ← Feature scaler
│   ├── feature_cols.pkl             ← Feature column order
│   ├── anomaly_features.pkl         ← Anomaly feature subset
│   ├── optimal_threshold.pkl        ← Tuned decision threshold
│   └── feature_importance.pkl       ← Feature importance ranking
├── api/
│   └── main.py                      ← FastAPI backend
├── dashboard/
│   └── app.py                       ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Why This Architecture?

| Question Judges Ask | Our Answer |
|---|---|
| Why synthetic data? | Kaggle only has PCA features. Real fraud uses device/geo/session signals. We generate these CONDITIONALLY on fraud labels using real fraud typologies (RBI, FATF). |
| Why LightGBM over XGBoost? | 2-5x faster inference, native imbalance handling, lower memory, histogram-based splitting. |
| Why multiple models? | LightGBM catches KNOWN patterns (supervised). IsoForest catches UNKNOWN anomalies (unsupervised). Together = fewer false negatives AND false positives. |
| Why fine-tune? | GridSearchCV with stratified 3-fold CV, optimising F1. Tests 128+ parameter combinations. |
| Why SHAP? | Regulatory compliance (RBI). Must explain WHY a transaction was blocked. |
| Why Trust Engine? | Traditional = score transactions. Ours = score CUSTOMERS continuously. Trust decays on suspicious activity, recovers slowly. |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| ML (Primary) | LightGBM | Fast, handles imbalance, feature importance |
| ML (Anomaly) | Isolation Forest | Unsupervised, catches novel fraud |
| Explainability | SHAP | Regulatory compliance, transparency |
| Trust | Dynamic Trust Engine | Continuous customer risk evaluation |
| API | FastAPI | Async, auto-docs, ML-friendly |
| Dashboard | Streamlit + Plotly | Real-time charts, Python-native |
| Blockchain | Hyperledger Fabric (sim) | Cross-bank fraud intelligence |
| Governance | STR Report Generator | RBI compliance |

---

## API Usage

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 85000,
    "hour": 3,
    "customer_id": "CUST-001",
    "new_device": true,
    "vpn_detected": true,
    "paste_detected": true,
    "failed_otp_count": 2,
    "device_trust_score": 20,
    "geo_distance_km": 2500
  }'
```

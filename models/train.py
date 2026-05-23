"""
FraudShield AI — Model Training Script
=======================================
Dataset : Kaggle Credit Card Fraud Detection
Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place    : fraud-demo/data/creditcard.csv

Trains two models:
  1. IsolationForest  — unsupervised anomaly detection
  2. XGBoost          — supervised fraud classification

Outputs  : models/iso_model.pkl  +  models/xgb_model.pkl
           models/scaler.pkl     +  models/feature_cols.pkl
"""

import os, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, precision_recall_curve)
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODEL_DIR  = os.path.dirname(__file__)

# ─────────────────────────────────────────────
# 1. Load & inspect data
# ─────────────────────────────────────────────
print("=" * 55)
print("  FraudShield AI — Training Pipeline")
print("=" * 55)

print("\n[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"      Rows: {len(df):,}  |  Fraud: {df['Class'].sum():,}  "
      f"({df['Class'].mean()*100:.2f}%)")

# ─────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────
print("\n[2/6] Engineering features...")

# Hour of day from 'Time' (seconds since first transaction)
df["Hour"] = (df["Time"] / 3600) % 24

# Amount log (reduces skew)
df["LogAmount"] = np.log1p(df["Amount"])

# Amount z-score (how unusual is this amount?)
df["AmountZscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

# Features to use
FEATURE_COLS = (
    [f"V{i}" for i in range(1, 29)]   # PCA features from dataset
    + ["LogAmount", "AmountZscore", "Hour"]
)

X = df[FEATURE_COLS]
y = df["Class"]

print(f"      Using {len(FEATURE_COLS)} features")

# ─────────────────────────────────────────────
# 3. Scale features
# ─────────────────────────────────────────────
print("\n[3/6] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# 4. Train IsolationForest (unsupervised)
# ─────────────────────────────────────────────
print("\n[4/6] Training Isolation Forest...")
iso = IsolationForest(
    n_estimators=200,
    contamination=0.002,   # ~0.17% fraud rate in dataset
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)
iso.fit(X_scaled)

# Anomaly score (more negative = more anomalous)
iso_scores = iso.decision_function(X_scaled)
# Normalise to 0-1 (1 = most anomalous)
iso_norm = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
print(f"      Anomaly score range: {iso_norm.min():.3f} – {iso_norm.max():.3f}")

# ─────────────────────────────────────────────
# 5. Train XGBoost (supervised)
# ─────────────────────────────────────────────
print("\n[5/6] Training XGBoost classifier...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Handle severe class imbalance
scale_pos = (y == 0).sum() / (y == 1).sum()
print(f"      Class imbalance ratio: {scale_pos:.0f}:1  (handled via scale_pos_weight)")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    use_label_encoder=False,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Evaluate
y_pred_prob = xgb.predict_proba(X_test)[:, 1]
y_pred      = (y_pred_prob > 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_prob)
cm  = confusion_matrix(y_test, y_pred)
print(f"\n      ROC-AUC  : {auc:.4f}")
print(f"      Confusion Matrix:")
print(f"        TN={cm[0,0]:,}  FP={cm[0,1]}")
print(f"        FN={cm[1,0]}   TP={cm[1,1]}")
print("\n" + classification_report(y_test, y_pred,
      target_names=["Legitimate", "Fraud"]))

# ─────────────────────────────────────────────
# 6. Save all artifacts
# ─────────────────────────────────────────────
print("\n[6/6] Saving model artifacts...")

joblib.dump(iso,          os.path.join(MODEL_DIR, "iso_model.pkl"))
joblib.dump(xgb,          os.path.join(MODEL_DIR, "xgb_model.pkl"))
joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))

print("      iso_model.pkl   ✓")
print("      xgb_model.pkl   ✓")
print("      scaler.pkl      ✓")
print("      feature_cols.pkl ✓")

print("\n" + "=" * 55)
print("  Training complete! Run the API next:")
print("  cd api && uvicorn main:app --reload")
print("=" * 55 + "\n")

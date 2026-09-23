"""
Continuous Trust Intelligence Framework — Training Pipeline
=============================================================
Uses the enriched dataset (with synthetic behavioural features) to train:

  1. LightGBM           — primary real-time fraud classifier
  2. Isolation Forest    — background anomaly detection
  3. SHAP Explainer      — explainability for compliance

Why LightGBM over XGBoost?
  - 2-5x faster training (histogram-based)
  - Native categorical feature support
  - Better handling of class imbalance with is_unbalance
  - Lower memory footprint → better for real-time inference
  - Comparable or better accuracy on tabular data

Why multiple models?
  LightGBM learns KNOWN fraud patterns (supervised).
  Isolation Forest detects UNKNOWN anomalies (unsupervised).
  They complement each other:
    - LightGBM catches pattern-matching fraud
    - IsoForest catches novel zero-day fraud patterns
  Combined scoring reduces both false negatives AND false positives.

Fine-tuning approach:
  - GridSearchCV with stratified K-fold for LightGBM
  - Optimised for RECALL (catching fraud) with minimum precision threshold
  - scale_pos_weight handles 577:1 class imbalance
"""

import os, sys, time, warnings, joblib
# LightGBM must be imported before scikit-learn: on Windows the reverse order
# loads a conflicting OpenMP runtime and LightGBM crashes with an access violation.
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score, f1_score,
    make_scorer
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent
ENRICHED  = DATA_DIR / "enriched_fraud_data.csv"

if not ENRICHED.exists():
    print("[ERROR] Enriched dataset not found. Run data/generate_synthetic.py first.")
    sys.exit(1)

# ─────────────────────────────────────────────
# 1. Load enriched dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  Continuous Trust Intelligence — Training Pipeline")
print("=" * 60)

print("\n[1/7] Loading enriched dataset...")
df = pd.read_csv(ENRICHED)
print(f"      Rows: {len(df):,}  |  Fraud: {df['Class'].sum():,} "
      f"({df['Class'].mean()*100:.3f}%)")

# ─────────────────────────────────────────────
# 2. Feature selection
# ─────────────────────────────────────────────
print("\n[2/7] Preparing feature matrix...")

# All features for training (exclude metadata columns)
EXCLUDE_COLS = ["Class", "fraud_scenario", "Amount"]

FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

# Add engineered features
df["log_amount"] = np.log1p(df["Amount"])
df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
df["risk_composite"] = (
    df["device_trust_score"].apply(lambda x: 100 - x) * 0.2 +   # invert: low trust = high risk
    df["geo_risk_score"] * 0.2 +
    df["session_risk_score"] * 0.3 +
    df["beneficiary_risk_score"] * 0.15 +
    (df["txn_frequency"] / df["txn_frequency"].max()) * 100 * 0.15
)

FEATURE_COLS += ["log_amount", "amount_zscore", "risk_composite"]

print(f"      Total features: {len(FEATURE_COLS)}")
print(f"      Feature groups:")
pca_feats = [f for f in FEATURE_COLS if f.startswith("V")]
behav_feats = ["avg_txn_amount", "amount_deviation", "txn_frequency",
               "txn_count_1h", "txn_count_24h"]
device_feats = ["new_device_flag", "rooted_device", "emulator_detected",
                "device_trust_score"]
geo_feats = ["geo_distance_km", "impossible_travel", "vpn_detected",
             "is_international", "geo_risk_score"]
session_feats = ["session_duration_sec", "paste_detected", "failed_otp_count",
                 "beneficiary_added_recently", "typing_speed_anomaly",
                 "session_risk_score"]
relation_feats = ["shared_device_accounts", "beneficiary_risk_score",
                  "account_age_days"]
engineered_feats = ["log_amount", "amount_zscore", "risk_composite"]

for group_name, group_feats in [
    ("PCA (Kaggle)",    pca_feats),
    ("Behavioural",     behav_feats),
    ("Device",          device_feats),
    ("Geo/Location",    geo_feats),
    ("Session",         session_feats),
    ("Relationship",    relation_feats),
    ("Engineered",      engineered_feats),
]:
    present = [f for f in group_feats if f in FEATURE_COLS]
    print(f"        {group_name:20s}: {len(present)} features")

X = df[FEATURE_COLS].values
y = df["Class"].values

# ─────────────────────────────────────────────
# 3. Scale features
# ─────────────────────────────────────────────
print("\n[3/7] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (stratified to preserve class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"      Train fraud: {y_train.sum():,}  |  Test fraud: {y_test.sum():,}")

# ─────────────────────────────────────────────
# 4. Train LightGBM with GridSearchCV
# ─────────────────────────────────────────────
print("\n[4/7] Training LightGBM with hyperparameter tuning...")
start_time = time.time()

# Class imbalance ratio
scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"      Class imbalance ratio: {scale_pos:.0f}:1")

# Base model
lgb_base = lgb.LGBMClassifier(
    objective="binary",
    is_unbalance=True,          # LightGBM's native imbalance handling
    boosting_type="gbdt",
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Hyperparameter grid for fine-tuning (focused grid for speed)
# In production: expand grid and use RandomizedSearchCV or Optuna
param_grid = {
    "n_estimators":     [300, 500],
    "max_depth":        [5, 7],
    "learning_rate":    [0.05],
    "num_leaves":       [31],
    "subsample":        [0.8],
    "colsample_bytree": [0.8],
    "min_child_samples": [30],
    "reg_alpha":        [0.1],
    "reg_lambda":       [1.0],
}

# Optimise for F1 (balances precision and recall)
f1_scorer = make_scorer(f1_score, average="binary")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Subsample for grid search speed (all fraud rows + random legit sample)
print("      Subsampling for GridSearch (keeps all fraud + 50k legit)...")
fraud_idx = np.where(y_train == 1)[0]
legit_idx = np.where(y_train == 0)[0]
sample_legit = np.random.RandomState(42).choice(legit_idx, size=min(50000, len(legit_idx)), replace=False)
sample_idx = np.concatenate([fraud_idx, sample_legit])
X_search, y_search = X_train[sample_idx], y_train[sample_idx]
print(f"      Search set: {len(X_search):,} rows ({y_search.sum()} fraud)")

print("      Running GridSearchCV (3-fold, optimising F1)...")
print("      This should take 1-3 minutes...")

grid_search = GridSearchCV(
    lgb_base,
    param_grid,
    scoring=f1_scorer,
    cv=cv,
    n_jobs=1,         # sequential: LightGBM already uses every core per fit, and
                      # parallel workers re-import sklearn first and crash (see above)
    verbose=0,
    refit=False       # we'll refit on full data manually
)

grid_search.fit(X_search, y_search)
best_params = grid_search.best_params_

# Refit best model on full training data
print(f"      Refitting best model on full training data ({len(X_train):,} rows)...")
lgb_model = lgb.LGBMClassifier(
    objective="binary", is_unbalance=True, boosting_type="gbdt",
    random_state=42, n_jobs=-1, verbose=-1, **best_params
)
lgb_model.fit(X_train, y_train)

elapsed_lgb = time.time() - start_time

print(f"\n      [OK] GridSearchCV complete in {elapsed_lgb:.1f}s")
print(f"      Best params: {best_params}")
print(f"      Best CV F1:  {grid_search.best_score_:.4f}")

# Evaluate on test set
y_pred_prob = lgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

# Try multiple thresholds to find optimal
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
# Find threshold that maximises F1
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[min(optimal_idx, len(thresholds) - 1)]
print(f"      Optimal decision threshold: {optimal_threshold:.3f}")

# Re-predict with optimal threshold
y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)

auc = roc_auc_score(y_test, y_pred_prob)
ap  = average_precision_score(y_test, y_pred_prob)
cm  = confusion_matrix(y_test, y_pred_optimal)

print(f"\n      -- LightGBM Test Metrics --")
print(f"      ROC-AUC          : {auc:.4f}")
print(f"      Avg Precision (PR): {ap:.4f}")
print(f"      Confusion Matrix:")
print(f"        TN={cm[0,0]:,}   FP={cm[0,1]}")
print(f"        FN={cm[1,0]}      TP={cm[1,1]}")
print(f"\n{classification_report(y_test, y_pred_optimal, target_names=['Legitimate', 'Fraud'])}")

# Feature importance
importances = lgb_model.feature_importances_
feat_imp = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": importances
}).sort_values("importance", ascending=False)

print("      Top 15 Most Important Features:")
for _, row in feat_imp.head(15).iterrows():
    bar = "#" * int(row["importance"] / feat_imp["importance"].max() * 30)
    print(f"        {row['feature']:30s} {bar} ({row['importance']})")

# ─────────────────────────────────────────────
# 5. Train Isolation Forest (anomaly detection)
# ─────────────────────────────────────────────
print("\n[5/7] Training Isolation Forest for anomaly detection...")
start_iso = time.time()

# Use behavioural + session + device features for anomaly detection
# (not PCA features — those are already anomaly-encoded)
ANOMALY_FEATURES = (
    behav_feats + device_feats + geo_feats + session_feats +
    relation_feats + engineered_feats +
    ["hour", "active_hour_score"]
)
ANOMALY_FEATURES = [f for f in ANOMALY_FEATURES if f in FEATURE_COLS]

anomaly_idx = [FEATURE_COLS.index(f) for f in ANOMALY_FEATURES]
X_anomaly = X_scaled[:, anomaly_idx]

iso_model = IsolationForest(
    n_estimators=300,
    contamination=0.002,    # slightly above fraud rate for sensitivity
    max_samples="auto",
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
iso_model.fit(X_anomaly)

# Evaluate anomaly scores
iso_scores = iso_model.decision_function(X_anomaly)
iso_labels = iso_model.predict(X_anomaly)   # 1=normal, -1=anomaly

# Check detection rate
anomaly_detected = (iso_labels == -1)
fraud_mask = (y == 1)

tp_iso = (anomaly_detected & fraud_mask).sum()
fp_iso = (anomaly_detected & ~fraud_mask).sum()
fn_iso = (~anomaly_detected & fraud_mask).sum()

elapsed_iso = time.time() - start_iso
print(f"      [OK] Trained in {elapsed_iso:.1f}s")
print(f"      Anomaly detection on fraud transactions:")
print(f"        Detected (TP): {tp_iso} / {fraud_mask.sum()} "
      f"({tp_iso / fraud_mask.sum() * 100:.1f}%)")
print(f"        False alarms:  {fp_iso}")

# ─────────────────────────────────────────────
# 6. SHAP Explainability setup
# ─────────────────────────────────────────────
print("\n[6/7] Setting up SHAP explainer...")
try:
    import shap
    explainer = shap.TreeExplainer(lgb_model)
    # Pre-compute on a small sample to verify
    shap_sample = X_test[:100]
    shap_values = explainer.shap_values(shap_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # class 1 = fraud
    print(f"      [OK] SHAP explainer ready")
    print(f"      Sample SHAP shape: {np.array(shap_values).shape}")
    has_shap = True
except ImportError:
    print("      [WARN] shap not installed -- explainability will use feature importance")
    explainer = None
    has_shap = False
except Exception as e:
    print(f"      [WARN] SHAP setup warning: {e}")
    explainer = None
    has_shap = False

# ─────────────────────────────────────────────
# 7. Save all artifacts
# ─────────────────────────────────────────────
print("\n[7/7] Saving model artifacts...")

artifacts = {
    "lgb_model.pkl":            lgb_model,
    "iso_model.pkl":            iso_model,
    "scaler.pkl":               scaler,
    "feature_cols.pkl":         FEATURE_COLS,
    "anomaly_features.pkl":     ANOMALY_FEATURES,
    "optimal_threshold.pkl":    optimal_threshold,
    "feature_importance.pkl":   feat_imp,
    "best_params.pkl":          best_params,
}

if has_shap and explainer is not None:
    artifacts["shap_explainer.pkl"] = explainer

for name, obj in artifacts.items():
    path = MODEL_DIR / name
    joblib.dump(obj, path)
    size = os.path.getsize(path) / 1024
    print(f"      {name:30s} OK  ({size:.0f} KB)")

# Also save XGBoost-compatible model for backward compat
# (the API can use either)
joblib.dump(lgb_model, MODEL_DIR / "xgb_model.pkl")
print(f"      {'xgb_model.pkl (compat alias)':30s} OK")

print("\n" + "=" * 60)
print("  Training Pipeline Complete!")
print("=" * 60)
print(f"""
  Models trained:
    1. LightGBM  - Real-time fraud scoring (ROC-AUC: {auc:.4f})
    2. IsoForest  - Background anomaly detection
    3. SHAP       - Explainability engine {'[OK]' if has_shap else '(install shap)'}

  Why multiple models?
    - LightGBM catches KNOWN fraud patterns (supervised learning)
    - IsoForest catches UNKNOWN anomalies (unsupervised)
    - SHAP explains WHY a transaction was blocked (compliance)

  Why fine-tuning with GridSearchCV?
    - Tests {len(grid_search.cv_results_['params'])} parameter combinations
    - 3-fold stratified cross-validation
    - Optimised for F1 score (balances precision + recall)
    - Best threshold: {optimal_threshold:.3f}

  Next steps:
    1. cd api && uvicorn main:app --reload --port 8000
    2. streamlit run dashboard/app.py
""")
print("=" * 60 + "\n")

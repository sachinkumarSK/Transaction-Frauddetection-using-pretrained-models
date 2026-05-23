"""
Continuous Trust Intelligence Framework — Synthetic Data Generator
===================================================================
Takes the Kaggle Credit Card Fraud dataset and enriches it with
realistic behavioural, device, geo, session, velocity, and
relationship features.

Outputs:
  data/enriched_fraud_data.csv  (ready for model training)

Why synthetic features?
  The Kaggle dataset contains only PCA-transformed features (V1-V28),
  Amount, and Time.  Real banking fraud systems use dozens of
  contextual signals.  We SYNTHESISE these because:
    1. Real bank data is confidential / not publicly available.
    2. The PATTERNS we encode (VPN + new device + impossible travel)
       are based on actual fraud typologies published by RBI, FATF,
       and bank fraud investigation reports.
    3. The synthetic generation is CONDITIONAL on the fraud label,
       so fraudulent transactions realistically exhibit suspicious
       behavioural combinations.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

DATA_DIR = Path(__file__).parent
KAGGLE_PATH = DATA_DIR / "creditcard.csv"
OUTPUT_PATH = DATA_DIR / "enriched_fraud_data.csv"

# ─────────────────────────────────────────────
# 1. Load base Kaggle dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  Continuous Trust Intelligence — Synthetic Data Generator")
print("=" * 60)

print("\n[1/8] Loading Kaggle Credit Card Fraud dataset...")
df = pd.read_csv(KAGGLE_PATH)
print(f"      Rows: {len(df):,}  |  Fraud: {df['Class'].sum():,} "
      f"({df['Class'].mean()*100:.3f}%)")

is_fraud = df["Class"] == 1
n = len(df)

# ─────────────────────────────────────────────
# 2. Time-based features
# ─────────────────────────────────────────────
print("\n[2/8] Generating time-based features...")

df["hour"] = (df["Time"] / 3600).astype(int) % 24
df["day_of_week"] = np.random.randint(0, 7, n)  # not in original data

# Active hour score: 1.0 if txn happens during user's typical hours
# Fraud transactions more likely to happen at unusual hours
df["active_hour_score"] = np.where(
    is_fraud,
    np.clip(np.random.beta(2, 5, n), 0, 1),       # fraud: low scores (unusual timing)
    np.clip(np.random.beta(7, 2, n), 0, 1)         # legit: high scores (normal timing)
)

# ─────────────────────────────────────────────
# 3. Behavioural features
# ─────────────────────────────────────────────
print("\n[3/8] Generating behavioural features...")

# Average transaction amount (user baseline) — fraud amounts deviate more
user_baseline = np.abs(np.random.normal(5000, 3000, n))
df["avg_txn_amount"] = user_baseline
df["amount_deviation"] = df["Amount"] / np.maximum(df["avg_txn_amount"], 1.0)

# Transaction frequency (txns/hour) — mule accounts have burst patterns
df["txn_frequency"] = np.where(
    is_fraud,
    np.clip(np.random.exponential(5, n), 0.5, 30),    # fraud: high burst
    np.clip(np.random.exponential(1.2, n), 0.1, 8)    # legit: normal
)

# Velocity: transactions in last 1 hour
df["txn_count_1h"] = np.where(
    is_fraud,
    np.random.choice([3, 5, 7, 10, 15], n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    np.random.choice([1, 1, 2, 2, 3], n)
)

# Velocity: transactions in last 24 hours
df["txn_count_24h"] = np.where(
    is_fraud,
    np.random.choice([5, 10, 15, 20, 30], n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    np.random.choice([1, 2, 3, 4, 5], n)
)

# ─────────────────────────────────────────────
# 4. Device features
# ─────────────────────────────────────────────
print("\n[4/8] Generating device features...")

# New device flag — fraudsters often use new/unknown devices
df["new_device_flag"] = np.where(
    is_fraud,
    np.random.choice([0, 1], n, p=[0.25, 0.75]),   # 75% fraud on new device
    np.random.choice([0, 1], n, p=[0.92, 0.08])    # 8% legit on new device
)

# Rooted/jailbroken device
df["rooted_device"] = np.where(
    is_fraud,
    np.random.choice([0, 1], n, p=[0.5, 0.5]),
    np.random.choice([0, 1], n, p=[0.97, 0.03])
)

# Emulator detected
df["emulator_detected"] = np.where(
    is_fraud,
    np.random.choice([0, 1], n, p=[0.6, 0.4]),
    np.random.choice([0, 1], n, p=[0.995, 0.005])
)

# Device trust score (0-100, composite of device age, known fingerprint, etc.)
df["device_trust_score"] = np.where(
    is_fraud,
    np.clip(np.random.normal(25, 15, n), 0, 100),
    np.clip(np.random.normal(82, 12, n), 0, 100)
)

# ─────────────────────────────────────────────
# 5. Geo features
# ─────────────────────────────────────────────
print("\n[5/8] Generating geo/location features...")

# Geo distance from last transaction (km) — impossible travel detection
df["geo_distance_km"] = np.where(
    is_fraud,
    np.clip(np.random.exponential(800, n), 0, 15000),  # fraud: large jumps
    np.clip(np.random.exponential(15, n), 0, 200)       # legit: small moves
)

# Impossible travel flag: can't travel 500+ km in < 1 hour
df["impossible_travel"] = np.where(
    (df["geo_distance_km"] > 500) & (df["txn_count_1h"] > 1),
    1, 0
)

# VPN / Proxy detected
df["vpn_detected"] = np.where(
    is_fraud,
    np.random.choice([0, 1], n, p=[0.35, 0.65]),   # 65% fraud uses VPN
    np.random.choice([0, 1], n, p=[0.96, 0.04])    # 4% legit uses VPN
)

# International transaction
df["is_international"] = np.where(
    is_fraud,
    np.random.choice([0, 1], n, p=[0.55, 0.45]),
    np.random.choice([0, 1], n, p=[0.93, 0.07])
)

# Geo risk score (country risk rating, 0-100)
df["geo_risk_score"] = np.where(
    is_fraud,
    np.clip(np.random.normal(65, 20, n), 0, 100),
    np.clip(np.random.normal(15, 10, n), 0, 100)
)

# ─────────────────────────────────────────────
# 6. Session features (THIS IS INNOVATION)
# ─────────────────────────────────────────────
print("\n[6/8] Generating session features...")

# Session duration on payment page (seconds)
# Fraudsters are very quick (automated) or very slow (hesitant)
df["session_duration_sec"] = np.where(
    is_fraud,
    np.where(np.random.random(n) > 0.5,
             np.random.uniform(2, 8, n),         # bot-fast
             np.random.uniform(300, 900, n)),     # suspicious slow
    np.random.uniform(15, 120, n)                 # normal user
)

# Paste detected (UPI ID / account number pasted, not typed)
df["paste_detected"] = np.where(
    is_fraud,
    np.random.choice([0, 1], n, p=[0.3, 0.7]),
    np.random.choice([0, 1], n, p=[0.85, 0.15])
)

# Failed OTP count in session
df["failed_otp_count"] = np.where(
    is_fraud,
    np.random.choice([0, 1, 2, 3, 4], n, p=[0.2, 0.25, 0.25, 0.2, 0.1]),
    np.random.choice([0, 0, 0, 1, 0], n)
)

# Beneficiary added recently (< 1 hour before large txn)
df["beneficiary_added_recently"] = np.where(
    is_fraud,
    np.random.choice([0, 1], n, p=[0.4, 0.6]),
    np.random.choice([0, 1], n, p=[0.95, 0.05])
)

# Typing speed anomaly score (deviation from user's normal typing pattern)
df["typing_speed_anomaly"] = np.where(
    is_fraud,
    np.clip(np.random.normal(0.75, 0.15, n), 0, 1),
    np.clip(np.random.normal(0.15, 0.10, n), 0, 1)
)

# Session risk score (composite)
df["session_risk_score"] = (
    df["paste_detected"] * 20 +
    df["failed_otp_count"] * 15 +
    df["beneficiary_added_recently"] * 20 +
    df["typing_speed_anomaly"] * 30 +
    np.where(df["session_duration_sec"] < 10, 15, 0)
).clip(0, 100)

# ─────────────────────────────────────────────
# 7. Relationship / Network features
# ─────────────────────────────────────────────
print("\n[7/8] Generating relationship features...")

# Shared device accounts (same device used by multiple accounts = mule ring)
df["shared_device_accounts"] = np.where(
    is_fraud,
    np.random.choice([1, 2, 3, 5, 8], n, p=[0.25, 0.25, 0.2, 0.2, 0.1]),
    np.random.choice([1, 1, 1, 1, 2], n)
)

# Beneficiary risk score (receiver account suspicion level)
df["beneficiary_risk_score"] = np.where(
    is_fraud,
    np.clip(np.random.normal(70, 15, n), 0, 100),
    np.clip(np.random.normal(10, 8, n), 0, 100)
)

# Account age (days) — new accounts are riskier
df["account_age_days"] = np.where(
    is_fraud,
    np.random.choice([1, 3, 7, 14, 30], n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    np.random.choice([30, 90, 180, 365, 730], n, p=[0.1, 0.15, 0.25, 0.3, 0.2])
)

# ─────────────────────────────────────────────
# 8. Create fraud scenario labels
# ─────────────────────────────────────────────
print("\n[8/8] Labelling fraud scenarios...")

# Tag specific fraud scenarios for explainability
def classify_fraud_scenario(row):
    if row["Class"] == 0:
        return "legitimate"
    flags = []
    if row["impossible_travel"]:
        flags.append("impossible_travel")
    if row["new_device_flag"] and row["vpn_detected"]:
        flags.append("account_takeover")
    if row["shared_device_accounts"] >= 3:
        flags.append("mule_account")
    if row["beneficiary_added_recently"] and row["amount_deviation"] > 5:
        flags.append("social_engineering")
    if row["emulator_detected"] or row["session_duration_sec"] < 10:
        flags.append("bot_attack")
    if row["failed_otp_count"] >= 2:
        flags.append("credential_stuffing")
    if not flags:
        flags.append("suspicious_pattern")
    return "|".join(flags)

df["fraud_scenario"] = df.apply(classify_fraud_scenario, axis=1)

# Print scenario distribution for fraud cases
fraud_df = df[df["Class"] == 1]
print("\n  Fraud Scenario Distribution:")
# Note: social_engineering scenario also generated
scenarios = fraud_df["fraud_scenario"].str.split("|").explode().value_counts()
for scenario, count in scenarios.items():
    print(f"    {scenario:30s} -> {count:,}")

# ─────────────────────────────────────────────
# Save enriched dataset
# ---------------------------------------------
# Drop the raw 'Time' column (we have 'hour' now)
df_out = df.drop(columns=["Time"])

df_out.to_csv(OUTPUT_PATH, index=False)
print(f"\n[OK] Enriched dataset saved to: {OUTPUT_PATH}")
print(f"   Rows: {len(df_out):,}  |  Features: {len(df_out.columns)}")
print(f"   File size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")

# Feature summary
print("\n" + "=" * 60)
print("  FEATURE SUMMARY")
print("=" * 60)
feature_groups = {
    "PCA (original)":     [f"V{i}" for i in range(1, 29)],
    "Time":               ["hour", "day_of_week", "active_hour_score"],
    "Behavioural":        ["avg_txn_amount", "amount_deviation", "txn_frequency",
                           "txn_count_1h", "txn_count_24h"],
    "Device":             ["new_device_flag", "rooted_device", "emulator_detected",
                           "device_trust_score"],
    "Geo/Location":       ["geo_distance_km", "impossible_travel", "vpn_detected",
                           "is_international", "geo_risk_score"],
    "Session":            ["session_duration_sec", "paste_detected", "failed_otp_count",
                           "beneficiary_added_recently", "typing_speed_anomaly",
                           "session_risk_score"],
    "Relationship":       ["shared_device_accounts", "beneficiary_risk_score",
                           "account_age_days"],
}

total_features = 0
for group, features in feature_groups.items():
    total_features += len(features)
    print(f"  {group:20s} : {len(features)} features")
print(f"  {'TOTAL':20s} : {total_features} features + Amount + Class")
print("=" * 60 + "\n")

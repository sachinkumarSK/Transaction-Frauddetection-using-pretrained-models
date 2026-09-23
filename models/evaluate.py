"""
PaymentGuardian — Model Evaluation & Leakage Audit
====================================================
Answers the questions an examiner will ask about the fraud model, WITHOUT
touching the production models used by the API (no .pkl file is overwritten).

  A. Split check      — random stratified split vs. temporal split (train on
                        the past, test on the future — how a bank deploys it).
  B. Feature sets     — Kaggle-only baseline vs. synthetic-only vs. full set.
                        The gap between baseline and full is the contribution
                        of the behavioural enrichment.
  C. Group contribution — add ONE synthetic group to the baseline, and remove
                        ONE group from the full set.
  D. Model comparison — Logistic Regression, Random Forest, XGBoost, LightGBM
                        (production config and without re-weighting),
                        Isolation Forest (unsupervised) on the same split.
  E. Leakage audit    — ROC-AUC of every feature ON ITS OWN. A synthetic feature
                        that beats the strongest REAL (Kaggle) feature is a
                        leakage flag.
  F. Signal stress    — progressively replace synthetic values with label-
                        independent noise to see how the model degrades if the
                        real-world signals are weaker than the simulated ones.
  G. Calibration      — Brier score + reliability curve.
  H. Seed stability   — same model, three random seeds: is the result stable?

Metrics: ROC-AUC, PR-AUC (primary — fraud is 0.17 %), precision/recall/F1 at a
threshold chosen on a validation slice, recall at 0.1 % false-positive rate,
and a money-based cost (missed fraud amount + analyst review cost per alert).

Run:   python models/evaluate.py            (full run, ~10 min)
       python models/evaluate.py --quick    (subsamples legit rows, ~1-2 min)
Output: models/reports/evaluation_report.md, evaluation_results.json, *.png
"""

import sys, json, time, argparse, warnings
from pathlib import Path

# LightGBM must be imported before scikit-learn: on Windows the reverse order
# loads a conflicting OpenMP runtime and LightGBM crashes with an access violation.
import lightgbm as lgb

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, brier_score_loss,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent.parent / "data"
MODEL_DIR  = Path(__file__).parent
REPORT_DIR = MODEL_DIR / "reports"
ENRICHED   = DATA_DIR / "enriched_fraud_data.csv"
KAGGLE     = DATA_DIR / "creditcard.csv"

SEED        = 42
REVIEW_COST = 5.0      # cost of one analyst review / false alarm (same currency as Amount)
TARGET_FPR  = 0.001    # operating point: 0.1 % of legit payments flagged

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="keep all fraud rows but only 25%% of legit rows")
args = parser.parse_args()

# ─────────────────────────────────────────────
# 1. Load data (+ real Time column for the temporal split)
# ─────────────────────────────────────────────
print("=" * 60)
print("  PaymentGuardian — Model Evaluation & Leakage Audit")
print("=" * 60)

for p in (ENRICHED, KAGGLE):
    if not p.exists():
        sys.exit(f"Missing {p}. Run data/generate_synthetic.py first.")

print("\n[1/10] Loading data...")
df = pd.read_csv(ENRICHED)
raw = pd.read_csv(KAGGLE, usecols=["Time", "V1", "Class"])
# generate_synthetic.py keeps row order and only drops Time, so rows align 1:1.
if len(raw) != len(df) or not np.allclose(raw["V1"].values, df["V1"].values):
    sys.exit("Enriched rows do not align with creditcard.csv — regenerate the enriched data.")
df["Time"] = raw["Time"].values
del raw

# Same engineered features as train_pipeline.py
df["log_amount"] = np.log1p(df["Amount"])
df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
df["risk_composite"] = (
    (100 - df["device_trust_score"]) * 0.2 +
    df["geo_risk_score"] * 0.2 +
    df["session_risk_score"] * 0.3 +
    df["beneficiary_risk_score"] * 0.15 +
    (df["txn_frequency"] / df["txn_frequency"].max()) * 100 * 0.15
)

if args.quick:
    keep = (df["Class"] == 1) | (np.random.RandomState(SEED).rand(len(df)) < 0.25)
    df = df[keep].reset_index(drop=True)
    print("      --quick: legit rows subsampled to 25 %")
print(f"      Rows: {len(df):,}  |  Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

# ─────────────────────────────────────────────
# 2. Feature groups
# ─────────────────────────────────────────────
PCA_FEATS = [f"V{i}" for i in range(1, 29)]
GROUPS = {
    # real signals from the Kaggle data
    "Kaggle (V1-V28, amount, hour)": PCA_FEATS + ["log_amount", "amount_zscore", "hour"],
    # synthetic signals from generate_synthetic.py
    "Time (synthetic)":  ["day_of_week", "active_hour_score"],
    "Behavioural":       ["avg_txn_amount", "amount_deviation", "txn_frequency",
                          "txn_count_1h", "txn_count_24h"],
    "Device":            ["new_device_flag", "rooted_device", "emulator_detected",
                          "device_trust_score"],
    "Geo/Location":      ["geo_distance_km", "impossible_travel", "vpn_detected",
                          "is_international", "geo_risk_score"],
    "Session":           ["session_duration_sec", "paste_detected", "failed_otp_count",
                          "beneficiary_added_recently", "typing_speed_anomaly",
                          "session_risk_score"],
    "Relationship":      ["shared_device_accounts", "beneficiary_risk_score",
                          "account_age_days"],
    "Engineered":        ["risk_composite"],
}
KAGGLE_FEATS    = GROUPS["Kaggle (V1-V28, amount, hour)"]
SYNTHETIC_FEATS = [f for g, fs in GROUPS.items() if not g.startswith("Kaggle") for f in fs]
FULL_FEATS      = KAGGLE_FEATS + SYNTHETIC_FEATS

y      = df["Class"].values
amount = df["Amount"].values

# ─────────────────────────────────────────────
# 3. Splits
# ─────────────────────────────────────────────
print("\n[2/10] Building splits...")

def temporal_split(frac_test=0.2, frac_val=0.2):
    """Past → train/val, future → test. Val is the latest slice of the past."""
    order = np.argsort(df["Time"].values, kind="stable")
    n_test = int(len(order) * frac_test)
    past, test = order[:-n_test], order[-n_test:]
    n_val = int(len(past) * frac_val)
    return past[:-n_val], past[-n_val:], test

def random_split():
    idx = np.arange(len(df))
    past, test = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    tr, val = train_test_split(past, test_size=0.2, random_state=SEED, stratify=y[past])
    return tr, val, test

SPLITS = {"temporal": temporal_split(), "random": random_split()}
for name, (tr, va, te) in SPLITS.items():
    print(f"      {name:9s} train {len(tr):,} ({y[tr].sum()} fraud) | "
          f"val {len(va):,} ({y[va].sum()}) | test {len(te):,} ({y[te].sum()})")

# ─────────────────────────────────────────────
# 4. Models & metrics
# ─────────────────────────────────────────────
try:
    LGB_PARAMS = joblib.load(MODEL_DIR / "best_params.pkl")
except Exception:
    LGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, num_leaves=31,
                      subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
                      reg_alpha=0.1, reg_lambda=1.0)

# The production model uses is_unbalance=True (~577:1 up-weighting of fraud).
# On the leaky synthetic features that is harmless, but on the real Kaggle
# features it makes LightGBM unstable (results swing with the random seed —
# see section H). The data/feature experiments therefore use the same
# hyper-parameters WITHOUT class re-weighting; the imbalance is handled by the
# decision threshold, which is tuned on the validation slice. Both variants are
# compared in D and H.
LGB_PROD   = "LightGBM (production: is_unbalance)"
LGB_STABLE = "LightGBM (no reweighting)"
PRIMARY    = LGB_STABLE

def make_model(kind, pos_weight, seed=SEED):
    if kind in (LGB_PROD, LGB_STABLE):
        return lgb.LGBMClassifier(objective="binary", is_unbalance=(kind == LGB_PROD),
                                  random_state=seed, n_jobs=-1, verbose=-1, **LGB_PARAMS)
    if kind == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=pos_weight,
                             eval_metric="aucpr", random_state=seed, n_jobs=-1, verbosity=0)
    if kind == "Random Forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                                      random_state=seed, n_jobs=-1)
    if kind == "Logistic Regression":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=2000))
    raise ValueError(kind)

def metrics(y_true, prob, thr, amt):
    """Threshold-free + operating-point + money metrics for one test set."""
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, prob)
    fraud_amt   = amt[y_true == 1].sum()
    missed_amt  = amt[(y_true == 1) & (pred == 0)].sum()
    cost        = missed_amt + (fp + tp) * REVIEW_COST
    return {
        "roc_auc":   round(float(roc_auc_score(y_true, prob)), 4),
        "pr_auc":    round(float(average_precision_score(y_true, prob)), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall":    round(tp / max(tp + fn, 1), 4),
        "f1":        round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        # best recall achievable while flagging at most 0.1 % of legit payments
        "recall_at_0.1pct_fpr": round(float(tpr[fpr <= TARGET_FPR].max()), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "fraud_amount_caught_pct": round(100 * (1 - missed_amt / max(fraud_amt, 1e-9)), 1),
        "cost_saving_pct": round(100 * (1 - cost / max(fraud_amt, 1e-9)), 1),
    }

def best_f1_threshold(y_true, prob):
    p, r, t = precision_recall_curve(y_true, prob)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[min(np.argmax(f1[:-1]), len(t) - 1)]) if len(t) else 0.5

def run(feats, split="temporal", kind=PRIMARY, X_override=None, keep_prob=False, seed=SEED):
    """Fit on train, pick threshold on val, report on test."""
    tr, va, te = SPLITS[split]
    X = X_override if X_override is not None else df[feats].values
    t0 = time.time()
    if kind == "Isolation Forest":
        # Unsupervised: fit on the training rows without labels; higher = more anomalous.
        m = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1)
        m.fit(X[tr])
        score = lambda idx: -m.score_samples(X[idx])
    else:
        pos_w = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = make_model(kind, pos_w, seed)
        m.fit(X[tr], y[tr])
        score = lambda idx: m.predict_proba(X[idx])[:, 1]
    thr = best_f1_threshold(y[va], score(va))
    prob_te = score(te)
    res = metrics(y[te], prob_te, thr, amount[te])
    res["threshold"] = round(thr, 4)
    res["n_features"] = len(feats)
    res["fit_seconds"] = round(time.time() - t0, 1)
    if keep_prob:
        res["_prob"] = prob_te
    return res

RESULTS = {"config": {"quick": args.quick, "review_cost": REVIEW_COST, "target_fpr": TARGET_FPR,
                      "rows": int(len(df)), "fraud": int(y.sum()), "primary_model": PRIMARY,
                      "lgb_params": LGB_PARAMS}}

def show(label, r):
    print(f"      {label:44s} ROC {r['roc_auc']:.4f} | PR {r['pr_auc']:.4f} | "
          f"P {r['precision']:.3f} R {r['recall']:.3f} | R@0.1%FPR {r['recall_at_0.1pct_fpr']:.3f}")

# ─────────────────────────────────────────────
# A. Split check
# ─────────────────────────────────────────────
print(f"\n[3/10] A. Random vs temporal split ({PRIMARY})...")
RESULTS["A_split"] = {}
for fs_name, feats in [("Kaggle-only", KAGGLE_FEATS), ("Full", FULL_FEATS)]:
    for s in ("random", "temporal"):
        r = run(feats, split=s)
        RESULTS["A_split"][f"{fs_name}, {s} split"] = r
        show(f"{fs_name}, {s} split", r)

# ─────────────────────────────────────────────
# B. Feature sets
# ─────────────────────────────────────────────
print(f"\n[4/10] B. Feature sets ({PRIMARY}, temporal split)...")
RESULTS["B_feature_sets"] = {}
curves = {}
for name, feats in [("Kaggle-only baseline", KAGGLE_FEATS),
                    ("Synthetic-only", SYNTHETIC_FEATS),
                    ("Full (production set)", FULL_FEATS)]:
    r = run(feats, keep_prob=True)
    curves[name] = r.pop("_prob")
    RESULTS["B_feature_sets"][name] = r
    show(name, r)
base_pr = RESULTS["B_feature_sets"]["Kaggle-only baseline"]["pr_auc"]
full_pr = RESULTS["B_feature_sets"]["Full (production set)"]["pr_auc"]

# ─────────────────────────────────────────────
# C. Group contribution — add one group / remove one group
# ─────────────────────────────────────────────
print(f"\n[5/10] C1. Add ONE synthetic group to the Kaggle baseline ({PRIMARY}, temporal)...")
RESULTS["C_add_group"] = {}
for g, gfeats in GROUPS.items():
    if g.startswith("Kaggle"):
        continue
    r = run(KAGGLE_FEATS + gfeats)
    r["pr_auc_gain"] = round(r["pr_auc"] - base_pr, 4)
    RESULTS["C_add_group"][f"Kaggle + {g}"] = r
    show(f"Kaggle + {g}", r)

print(f"\n[6/10] C2. Remove ONE synthetic group from the full set ({PRIMARY}, temporal)...")
# risk_composite is built from Device/Geo/Session/Relationship/Behavioural,
# so it is dropped together with any of those groups to avoid a back door.
COMPOSITE_PARENTS = {"Behavioural", "Device", "Geo/Location", "Session", "Relationship"}
RESULTS["C_remove_group"] = {}
for g, gfeats in GROUPS.items():
    if g.startswith("Kaggle"):
        continue
    drop = set(gfeats) | ({"risk_composite"} if g in COMPOSITE_PARENTS else set())
    r = run([f for f in FULL_FEATS if f not in drop])
    r["pr_auc_drop"] = round(full_pr - r["pr_auc"], 4)
    RESULTS["C_remove_group"][f"Full - {g}"] = r
    show(f"Full - {g}", r)

# ─────────────────────────────────────────────
# D. Model comparison
# ─────────────────────────────────────────────
print("\n[7/10] D. Model comparison (temporal split)...")
RESULTS["D_models"] = {}
for fs_name, feats in [("Kaggle-only", KAGGLE_FEATS), ("Full", FULL_FEATS)]:
    RESULTS["D_models"][fs_name] = {}
    for kind in ("Logistic Regression", "Random Forest", "XGBoost", LGB_PROD, LGB_STABLE,
                 "Isolation Forest"):
        try:
            r = run(feats, kind=kind)
        except ImportError:
            print(f"      {kind}: not installed, skipped")
            continue
        RESULTS["D_models"][fs_name][kind] = r
        show(f"{fs_name:11s} {kind}", r)

# ─────────────────────────────────────────────
# E. Leakage audit — single-feature AUC
# ─────────────────────────────────────────────
print("\n[8/10] E. Leakage audit (each feature alone)...")
audit = []
for f in FULL_FEATS:
    auc = roc_auc_score(y, df[f].values)
    audit.append({"feature": f, "auc": round(max(auc, 1 - auc), 4),
                  "source": "kaggle" if f in KAGGLE_FEATS else "synthetic"})
audit.sort(key=lambda d: -d["auc"])
best_real = next(a for a in audit if a["source"] == "kaggle")
flagged = [a for a in audit if a["source"] == "synthetic" and a["auc"] > best_real["auc"]]
RESULTS["E_leakage_audit"] = {"strongest_real_feature": best_real,
                              "synthetic_above_strongest_real": [a["feature"] for a in flagged],
                              "features": audit}
print(f"      Strongest REAL feature: {best_real['feature']} (AUC {best_real['auc']:.4f})")
print(f"      {len(flagged)} synthetic features beat it on their own:")
for a in flagged:
    print(f"        {a['feature']:28s} {a['auc']:.4f}")

# ─────────────────────────────────────────────
# F. Signal stress test
# ─────────────────────────────────────────────
print(f"\n[9/10] F. Signal stress test ({PRIMARY}, temporal)...")
# For a fraction p of rows (independently per synthetic column) the value is
# replaced by the value of a random other row — i.e. label-independent noise.
# p=0 is the dataset as generated; p=0.9 keeps only 10 % of the simulated signal.
RESULTS["F_signal_stress"] = {}
rng = np.random.RandomState(SEED)
base_X = df[FULL_FEATS].values.astype(float)
syn_cols = [FULL_FEATS.index(f) for f in SYNTHETIC_FEATS]
for p in (0.0, 0.5, 0.8, 0.9, 0.95, 1.0):
    X = base_X.copy()
    for c in syn_cols:
        mask = rng.rand(len(X)) < p
        X[mask, c] = base_X[rng.randint(0, len(X), mask.sum()), c]
    r = run(FULL_FEATS, X_override=X)
    RESULTS["F_signal_stress"][f"{int(p*100)}% of synthetic values replaced"] = r
    show(f"{int(p*100)}% of synthetic values replaced", r)
print(f"      (Kaggle-only baseline PR-AUC for reference: {base_pr:.4f})")

# ─────────────────────────────────────────────
# G. Calibration + H. Seed stability
# ─────────────────────────────────────────────
print("\n[10/10] G. Calibration + H. seed stability (Kaggle-only, temporal)...")
te = SPLITS["temporal"][2]
fraud_te = y[te] == 1
RESULTS["G_calibration"] = {}
for name in ("Kaggle-only baseline", "Full (production set)"):
    prob = curves[name]
    RESULTS["G_calibration"][name] = {
        "brier": float(f"{brier_score_loss(y[te], prob):.3g}"),
        # share of real frauds the model is NOT sure about (0 % = it is always 0 or 1)
        "frauds_scored_between_1_and_99_pct":
            round(float(((prob[fraud_te] > 0.01) & (prob[fraud_te] < 0.99)).mean() * 100), 1),
    }

SEEDS = (42, 7, 2024)
RESULTS["H_seed_stability"] = {}
for kind in (LGB_PROD, LGB_STABLE, "XGBoost"):
    try:
        runs = [run(KAGGLE_FEATS, kind=kind, seed=s) for s in SEEDS]
    except ImportError:
        print(f"      {kind}: not installed, skipped")
        continue
    agg = {}
    for k in ("roc_auc", "pr_auc", "recall_at_0.1pct_fpr"):
        v = np.array([r_[k] for r_ in runs])
        agg[k] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                  "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}
    RESULTS["H_seed_stability"][kind] = agg
    print(f"      {kind:44s} PR-AUC {agg['pr_auc']['mean']:.4f} ± {agg['pr_auc']['std']:.4f} "
          f"(min {agg['pr_auc']['min']:.4f}, max {agg['pr_auc']['max']:.4f})")

# ─────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────
REPORT_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(6.4, 4.8))
for (name, prob), ls in zip(curves.items(), ("-", "--", ":")):
    p, r, _ = precision_recall_curve(y[te], prob)
    ax.plot(r, p, ls=ls, lw=2, label=f"{name} (PR-AUC {average_precision_score(y[te], prob):.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall — temporal test set")
ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(REPORT_DIR / "pr_curves.png", dpi=140); plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
groups = [g for g in GROUPS if not g.startswith("Kaggle")]
gains = [RESULTS["C_add_group"][f"Kaggle + {g}"]["pr_auc_gain"] for g in groups]
drops = [RESULTS["C_remove_group"][f"Full - {g}"]["pr_auc_drop"] for g in groups]
a1.barh(groups, gains, color="#1673c4")
a1.set_xlabel("PR-AUC gained over the Kaggle-only baseline"); a1.set_title("Add ONE group to the baseline")
a2.barh(groups, drops, color="#e5484d")
a2.set_xlabel("PR-AUC lost from the full set"); a2.set_title("Remove ONE group from the full set")
xmax = max(max(gains + drops), 0.05) * 1.3
for a, vals in ((a1, gains), (a2, drops)):
    a.set_xlim(min(0, min(vals)) * 1.3, xmax); a.grid(axis="x", alpha=.3)
    for i, v in enumerate(vals):
        a.text(max(v, 0) + xmax * 0.01, i, f"{v:+.3f}", va="center", fontsize=8)
a1.invert_yaxis()
fig.tight_layout(); fig.savefig(REPORT_DIR / "ablation.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 7))
top = audit[:25]
ax.barh([a["feature"] for a in top], [a["auc"] for a in top],
        color=["#e5484d" if a["source"] == "synthetic" else "#1673c4" for a in top])
ax.axvline(best_real["auc"], ls="--", c="k", lw=1); ax.set_xlim(0.5, 1.0)
ax.set_xlabel("ROC-AUC of the feature on its own (red = synthetic, blue = Kaggle)\n"
              f"dashed line = strongest real feature ({best_real['feature']}, {best_real['auc']:.3f})")
ax.set_title("Leakage audit — top 25 single features"); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(REPORT_DIR / "leakage_audit.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 5.2))
for name in ("Kaggle-only baseline", "Full (production set)"):
    frac, mean = calibration_curve(y[te], curves[name], n_bins=10, strategy="quantile")
    ax.plot(mean, frac, marker="o", label=name)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraud rate")
ax.set_title("Calibration — temporal test set"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(REPORT_DIR / "calibration.png", dpi=140); plt.close(fig)

# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────
(REPORT_DIR / "evaluation_results.json").write_text(json.dumps(RESULTS, indent=2))

def table(rows, cols):
    head = "| Setting | " + " | ".join(c for c, _ in cols) + " |"
    sep  = "|---|" + "---|" * len(cols)
    body = ["| " + name + " | " + " | ".join(str(r.get(k, "—")) for _, k in cols) + " |"
            for name, r in rows.items()]
    return "\n".join([head, sep, *body])

MAIN = [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Precision", "precision"),
        ("Recall", "recall"), ("F1", "f1"), ("Recall @0.1% FPR", "recall_at_0.1pct_fpr"),
        ("Fraud amount caught %", "fraud_amount_caught_pct"), ("Cost saving %", "cost_saving_pct")]
SHORT = [("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc"), ("Recall @0.1% FPR", "recall_at_0.1pct_fpr")]

md = [
    "# Model Evaluation Report",
    "",
    f"Generated by `models/evaluate.py`{' (--quick mode: 25 % of legit rows)' if args.quick else ''}. "
    f"Rows: {len(df):,}, fraud: {int(y.sum())}. Unless stated otherwise: **{PRIMARY}** "
    f"(production hyper-parameters; see H for why re-weighting is off), **temporal split** "
    f"(train on the first 64 % of time, threshold chosen on the next 16 %, tested on the last 20 %). "
    f"Cost saving = 1 − (missed fraud amount + {REVIEW_COST:g} per alert reviewed) / total fraud amount.",
    "",
    "## A. Random vs temporal split", "", table(RESULTS["A_split"], MAIN), "",
    "## B. Feature sets", "", table(RESULTS["B_feature_sets"], MAIN), "",
    "![PR curves](pr_curves.png)", "",
    "## C. Group contribution", "",
    "**C1 — add one synthetic group to the Kaggle-only baseline**", "",
    table(RESULTS["C_add_group"], SHORT + [("PR-AUC gain", "pr_auc_gain")]), "",
    "**C2 — remove one synthetic group from the full set**", "",
    table(RESULTS["C_remove_group"], SHORT + [("PR-AUC lost", "pr_auc_drop")]), "",
    "![Group contribution](ablation.png)", "",
    "## D. Model comparison", "",
]
for fs_name, rows in RESULTS["D_models"].items():
    md += [f"**{fs_name} features**", "", table(rows, MAIN + [("Fit (s)", "fit_seconds")]), ""]
md += [
    "## E. Leakage audit (each feature on its own)", "",
    f"Strongest real (Kaggle) feature: **{best_real['feature']}**, ROC-AUC {best_real['auc']}. "
    f"**{len(flagged)} synthetic features beat it on their own** — a real-world signal "
    f"this clean would be unusual, so these are leakage flags.", "",
    "| Feature | Source | Single-feature AUC |", "|---|---|---|",
    *[f"| {a['feature']} | {a['source']} | {a['auc']} |" for a in audit[:15]], "",
    "![Leakage audit](leakage_audit.png)", "",
    "## F. Signal stress test (synthetic values replaced by label-independent noise)", "",
    table(RESULTS["F_signal_stress"], SHORT + [("Precision", "precision"), ("Recall", "recall")]),
    "", f"Kaggle-only baseline PR-AUC for reference: {base_pr}", "",
    "## G. Calibration", "",
    "| Model | Brier score | % of test frauds scored between 1 % and 99 % |", "|---|---|---|",
    *[f"| {k} | {v['brier']} | {v['frauds_scored_between_1_and_99_pct']} |"
      for k, v in RESULTS["G_calibration"].items()], "",
    "![Calibration](calibration.png)", "",
    f"## H. Seed stability (Kaggle-only features, seeds {', '.join(map(str, SEEDS))})", "",
    "| Model | PR-AUC mean ± std | PR-AUC min – max | ROC-AUC mean ± std |", "|---|---|---|---|",
    *[f"| {k} | {v['pr_auc']['mean']} ± {v['pr_auc']['std']} | {v['pr_auc']['min']} – {v['pr_auc']['max']} "
      f"| {v['roc_auc']['mean']} ± {v['roc_auc']['std']} |" for k, v in RESULTS["H_seed_stability"].items()], "",
]
(REPORT_DIR / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")

print("\n" + "=" * 60)
print(f"  Done. Report: {REPORT_DIR / 'evaluation_report.md'}")
print("=" * 60)

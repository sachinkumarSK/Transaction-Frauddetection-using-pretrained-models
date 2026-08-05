# 🛡️ PayGuard — Real-Time Payment Fraud Protection

> PayGuard is a fraud-prevention engine that sits **on top of** a bank's existing
> systems as an intelligent **trust layer**. For every digital payment it checks
> the transaction, the device, the customer's history, the receiver's reputation,
> and a shared fraud memory — then produces a single **Risk Score (0–100)** and a
> decision: **Approved · Risky (verify) · Blocked**.
>
> Built for the Indian Overseas Bank (IOB) problem statement.
> The mock payment app is branded **"IOB Pay"**; the fraud engine behind it is **PayGuard**.

This README explains **everything** — the idea, the data, the models, every
feature, the database, the "blockchain", every API, and the whole user interface —
in plain language, so you can understand and present it confidently.

---

## 📑 Table of contents
1. [The problem & the idea](#1-the-problem--the-idea)
2. [Glossary — words you'll see](#2-glossary--words-youll-see)
3. [How it works (architecture)](#3-how-it-works-architecture)
4. [Step-by-step workflow](#4-step-by-step-workflow)
5. [Project structure](#5-project-structure)
6. [Tech stack](#6-tech-stack)
7. [The database (SQLite)](#7-the-database-sqlite)
8. [The dataset](#8-the-dataset)
9. [The features (all 57)](#9-the-features-all-57)
10. [The models](#10-the-models)
11. [The Risk Score (how the decision is made)](#11-the-risk-score-how-the-decision-is-made)
12. [Why the AI "fraud probability" looks like 0% or 100%](#12-why-the-ai-fraud-probability-looks-like-0-or-100)
13. [Risk Score vs Customer Trust (two different numbers)](#13-risk-score-vs-customer-trust)
14. [The "blockchain" — the honest truth](#14-the-blockchain--the-honest-truth)
15. [API reference](#15-api-reference)
16. [The dashboard (analyst console)](#16-the-dashboard-analyst-console)
17. [The IOB Pay simulator](#17-the-iob-pay-simulator)
18. [How to run it](#18-how-to-run-it)
19. [Demo script](#19-demo-script)
20. [Roadmap](#20-roadmap)
21. [FAQ](#21-faq)

---

## 1. The problem & the idea

**The problem.** Banks lose money to fraud that rule-based systems miss —
especially:
- **APP fraud** (Authorized Push Payment): the customer is *tricked* into sending
  money themselves. The login and OTP are all genuine, so it's hard to catch.
- **Mule accounts**: accounts used to receive and move stolen money.
- **First-time beneficiary scams**: money sent to a receiver never paid before.

Rules are rigid, cause false alarms, and have no memory or shared intelligence.

**The idea.** Don't replace the bank's engine — add a smart **trust layer**. For
each payment, PayGuard asks six questions:
1. Does the AI model think this transaction looks fraudulent?
2. Is the device trustworthy (new? rooted? emulator?)?
3. How has this customer behaved historically?
4. Is the **receiver** known-good or known-bad?
5. Did authentication go smoothly (OTP failures, pasted account numbers)?
6. Has our **shared fraud memory** seen this receiver/device before?

It blends those into **one Risk Score (0–100)** and decides. Crucially it
**learns**: when a fraud is blocked, the receiver/device is remembered, so the
**next** payment to that same account is caught instantly — even if it looks clean.

---

## 2. Glossary — words you'll see

| Term | Plain meaning |
|---|---|
| **Risk Score** | The final 0–100 score for one payment. Higher = riskier. (Internally the "Transaction Trust Index / TTI".) |
| **Customer Trust** | A 0–100 reputation for the *customer*, built over their history. Higher = more trusted. **Different from Risk Score.** |
| **APP fraud** | Victim is socially engineered into paying a scammer themselves. |
| **Mule account** | A bank account used to launder stolen money. |
| **Beneficiary** | The *receiver* of a payment. |
| **Reputation ledger** | Our shared memory of known-bad receivers/devices (risk score per entity). |
| **Feedback loop** | When a payment is Blocked, its receiver/device is remembered so future payments to it are caught. |
| **SHAP** | A method that explains *which features* drove the AI's decision (for compliance). |
| **Step-up / MFA** | For "Risky" payments, ask for an extra check (another OTP) instead of approving or blocking. |
| **Decision** | One of: **Approved**, **Risky** (verify with OTP), **Blocked**. |

---

## 3. How it works (architecture)

```
     ┌───────────────────────────────────────────────────────────┐
     │   IOB Pay app  /  UPI  /  Card  /  Wallet   (payment)      │
     └───────────────────────────┬───────────────────────────────┘
                                  │  HTTP POST /score  (JSON)
                                  ▼
     ┌───────────────────────────────────────────────────────────┐
     │              PayGuard Backend  (api/main.py)               │
     │                                                            │
     │  1) Build 57-feature vector  →  scale                      │
     │  ┌──────────────────────────────────────────────────────┐ │
     │  │ LightGBM        → fraud probability                  │ │
     │  │ Isolation Forest→ anomaly score                      │ │
     │  │ SHAP            → explanation                         │ │
     │  └──────────────────────────────────────────────────────┘ │
     │  2) Dynamic Trust Engine → Customer Trust                  │
     │  3) Reputation ledger lookup → receiver / device risk      │
     │  ┌──────────────────────────────────────────────────────┐ │
     │  │      RISK SCORE = weighted blend of 6 checks         │ │
     │  └──────────────────────────────────────────────────────┘ │
     │  4) Decision: Approved / Risky / Blocked                  │
     │  5) Ledger override (confirmed-fraud entity escalates)    │
     │  6) On Block → remember receiver/device (feedback loop)   │
     │  7) Save everything to SQLite  +  live feed               │
     └───────────────────────────┬───────────────────────────────┘
              ┌──────────────────┼───────────────────┐
              ▼                  ▼                    ▼
     ┌─────────────────┐  ┌──────────────┐   ┌──────────────────┐
     │ SQLite database │  │ Streamlit    │   │ IOB Pay app      │
     │ (durable state) │  │ dashboard    │   │ (React, /pay)    │
     └─────────────────┘  └──────────────┘   └──────────────────┘
```

There are **three programs**: the **PayGuard API** (the brain), the **Streamlit
dashboard** (what a fraud analyst sees), and the **IOB Pay app** (a mock payment
screen). The API also saves everything to a **SQLite database** so nothing is lost
on restart.

---

## 4. Step-by-step workflow

1. **Customer makes a payment** (from IOB Pay or any channel).
2. **Backend gathers details** — amount, time, device, location, session
   behaviour, receiver ID.
3. **Receiver reputation is checked** in the ledger.
4. **AI scores it** — LightGBM (fraud probability) + Isolation Forest (anomaly).
5. **Risk Score is computed** by blending 6 checks.
6. **Decision** — Approved / Risky (verify) / Blocked, with a ledger override.
7. **SHAP explains** the decision.
8. **On Block**, the receiver/device is remembered (feedback loop), a compliance
   **STR report** is generated, and everything is **saved to SQLite**.

---

## 5. Project structure

```
fraud-demo/
├── data/
│   ├── creditcard.csv            ← Kaggle base dataset (284,807 payments)
│   ├── generate_synthetic.py     ← Adds 25+ behavioural features → enriched CSV
│   ├── enriched_fraud_data.csv   ← Generated training data (57 features)
│   └── payguard.db               ← SQLite database (created automatically)
├── models/
│   ├── train_pipeline.py         ← Trains LightGBM + Isolation Forest + SHAP
│   ├── trust_engine.py           ← Dynamic Trust Engine (customer trust)
│   ├── reputation.py             ← Reputation ledger (receiver/device memory)
│   ├── db.py                     ← SQLite persistence layer
│   └── *.pkl                     ← Trained models + settings
├── api/
│   └── main.py                   ← PayGuard backend (FastAPI)
├── dashboard/
│   └── app.py                    ← Analyst dashboard (Streamlit)
├── iobpay/
│   └── index.html                ← IOB Pay simulator (React, served at /pay)
├── requirements.txt
└── README.md                     ← This file
```

**Who talks to whom:** `api/main.py` loads the models and imports
`trust_engine.py`, `reputation.py`, and `db.py`. The **dashboard** and the **IOB
Pay app** are pure clients — they only call the API over HTTP.

---

## 6. Tech stack

| Layer | Technology | Why |
|---|---|---|
| Main AI model | **LightGBM** | Fast, handles rare-fraud imbalance, explainable. |
| Anomaly model | **Isolation Forest** | Unsupervised — catches *novel* fraud. |
| Explainability | **SHAP** | Regulators require explaining *why* a payment was blocked. |
| Customer trust | **Dynamic Trust Engine** (custom) | Scores the customer over time, not just one payment. |
| Fraud memory | **Reputation ledger** (custom) | Shared receiver/device intelligence. |
| Backend API | **FastAPI** | Async, auto docs (`/docs`), input validation. |
| Database | **SQLite** (built-in `sqlite3`) | Durable storage, zero setup. |
| Dashboard | **Streamlit + Plotly** | Fast, Python-native, interactive charts. |
| Payment app | **React** (via CDN) | Single-file mock payment screen, no build step. |
| Data | **pandas + NumPy** | Feature engineering + synthetic data. |
| Model storage | **joblib** | Saves/loads trained models (`.pkl`). |

> **Production stack (roadmap):** React analyst console, PostgreSQL, Redis,
> Docker, real Hyperledger Fabric, Grafana. See §20.

---

## 7. The database (SQLite)

PayGuard uses a **SQLite database** at `data/payguard.db` (created automatically on
first run — no setup needed). It uses Python's **built-in** `sqlite3`, so there's
**no extra dependency** to install.

**Why:** without it, everything reset every time the API restarted. Now the things
PayGuard *learns* survive restarts. Three tables:

| Table | What it stores |
|---|---|
| `reputation` | The fraud memory — each known-bad receiver/device with its risk. |
| `customers` | Each customer's trust score and history counters. |
| `transactions` | Every scored payment (powers the live feed and analytics). |

**How it flows:** on startup the API loads the ledger, customer trust, and recent
transactions from SQLite. Every time a payment is scored or a fraud is remembered,
the change is written back. **Verified:** stop the API, restart it — the ledger,
customer trust, and feed are all still there.

To wipe everything and start fresh, just delete `data/payguard.db`.

> **Why SQLite and not PostgreSQL?** The PRD names PostgreSQL for large-scale
> production. SQLite gives the *same durability* with zero setup, which is the
> right choice at this stage. Swapping to PostgreSQL later is straightforward
> because all database code lives in one file (`models/db.py`).

---

## 8. The dataset

Two layers of data.

### 8a. Base: Kaggle Credit Card Fraud dataset
- **284,807 payments**, of which **492 are fraud (0.172 %)** — very imbalanced
  (~**577 legit : 1 fraud**).
- Columns: `Time`, `Amount`, `Class` (0/1 label), and **V1–V28** — anonymised
  **PCA-transformed** signals (we can't know what each means; they're compressed).

### 8b. Enrichment: synthetic behavioural features (`generate_synthetic.py`)
The Kaggle data has no device/location/session signals — but real fraud is caught
by exactly those. So we **create 25+ realistic features**, generated based on the
fraud label using published fraud patterns (RBI, FATF).

**Why this is legitimate, not cheating:** we don't put the answer in one magic
feature. We reproduce *real statistical patterns*. Example — `new_device_flag`:
fraud rows are on a new device 75 % of the time, legit rows only 8 %. That mirrors
reality (fraudsters use fresh devices), so the model learns a realistic signal.

Output: `enriched_fraud_data.csv` — the 284,807 rows with the full feature set.

---

## 9. The features (all 57)

The model uses **57 features**.

- **PCA (28):** `V1`–`V28` — anonymised signals from Kaggle.
- **Time (3):** `hour`, `day_of_week`, `active_hour_score` (is it the user's normal hours?).
- **Behavioural (5):** `avg_txn_amount`, `amount_deviation` (amount ÷ average),
  `txn_frequency`, `txn_count_1h`, `txn_count_24h`.
- **Device (4):** `new_device_flag`, `rooted_device`, `emulator_detected`,
  `device_trust_score`.
- **Location (5):** `geo_distance_km`, `impossible_travel` (moved >500 km in <1 h),
  `vpn_detected`, `is_international`, `geo_risk_score`.
- **Session (6):** `session_duration_sec`, `paste_detected` (account number pasted,
  not typed), `failed_otp_count`, `beneficiary_added_recently`,
  `typing_speed_anomaly`, `session_risk_score`.
- **Relationship (3):** `shared_device_accounts` (mule-ring signal),
  `beneficiary_risk_score`, `account_age_days`.
- **Engineered (3):** `log_amount`, `amount_zscore`, `risk_composite`.

**Most important features** (from the trained model): `beneficiary_risk_score` ›
`account_age_days` › `risk_composite` › `typing_speed_anomaly` › `txn_count_1h`.
In other words, **who you're paying and how new they are** matter most — exactly
the APP/mule fraud we target.

---

## 10. The models

### 10a. LightGBM — main fraud classifier
- **What:** a gradient-boosted decision-tree model that outputs a **fraud
  probability (0–1)**.
- **Why LightGBM:** fast inference, handles the 577:1 imbalance natively, explainable.
- **Tuned parameters** (found by GridSearchCV, 3-fold, optimising F1):

  ```
  n_estimators=300   learning_rate=0.05   max_depth=5   num_leaves=31
  subsample=0.8      colsample_bytree=0.8  min_child_samples=30
  reg_alpha=0.1      reg_lambda=1.0        is_unbalance=True
  ```

### 10b. Isolation Forest — anomaly detection
- **What:** an *unsupervised* model that scores how *unusual* a payment is,
  ignoring the label.
- **Why:** LightGBM only knows patterns it was trained on. Isolation Forest catches
  **new / never-seen** fraud shapes. Uses 28 behavioural features.

### 10c. SHAP — explainability
- **What:** for each decision, shows which features pushed the score **up** or
  **down**.
- **Why:** RBI compliance — you must justify a block.

### 10d. Dynamic Trust Engine (`trust_engine.py`)
- **What:** a **per-customer trust score (0–100)** that **drops** on suspicious
  activity and **recovers slowly** with good behaviour.
- **Why it's special:** most systems score *transactions*. This scores the
  **customer over time** — like a bank's real view of a client. Its score becomes
  the "Customer History" check inside the Risk Score.

### 10e. Reputation ledger (`reputation.py`)
- **What:** the **fraud memory** — a risk score (0–100) for each receiver and
  device, seeded with known-bad entities (e.g. `BENF-MULE-001`).
- **Methods:** `lookup` (read), `report_fraud` (write — the feedback loop),
  `observe`. Stores only IDs + risk, no personal data. Backed by SQLite.

---

## 11. The Risk Score (how the decision is made)

The Risk Score turns six checks into **one 0–100 number** (higher = riskier).

### The six checks and their weights
| # | Check | Weight | Comes from |
|---|---|---|---|
| 1 | **Transaction Risk** | 0.35 | LightGBM + Isolation Forest |
| 2 | **Device Trust** | 0.15 | Device signals |
| 3 | **Customer History** | 0.15 | Dynamic Trust Engine + amount/velocity |
| 4 | **Receiver Reputation** | 0.15 | Ledger lookup (first-time = premium) |
| 5 | **Authentication** | 0.10 | Failed OTPs + pasted account number |
| 6 | **Shared Fraud Memory** | 0.10 | Worst ledger risk of receiver/device |

**Risk Score = Σ (check × weight).** The dashboard shows every check's score and
contribution, so nothing is hidden.

### The decision
| Risk Score | Decision | Meaning |
|---|---|---|
| **Under 35** | ✅ **Approved** | Looks safe — let it through. |
| **35 – 70** | ⚠️ **Risky** | Not sure — ask for an extra OTP (step-up MFA). |
| **70+** | ⛔ **Blocked** | Almost certainly fraud — stop it. |

**Why the middle band asks for MFA:** it's the "unsure" zone. Blocking everyone
here annoys real customers (false alarms); approving everyone lets fraud through.
So we ask for one more proof of identity. **Not every payment is MFA** — only the
middle band. Safe payments are approved instantly; clear fraud is blocked outright.

### Ledger override (safety net)
A confirmed-fraud receiver/device **overrides** the score: ledger risk ≥ 70 forces
**Blocked**, ≥ 40 forces at least **Risky**. It never downgrades a decision.

### Feedback loop (learning)
On a **Block**, PayGuard remembers the receiver (+75) and device (+45) in the
ledger. So the **next** payment to that receiver is caught automatically — even if
it looks perfectly clean. (The lookup happens *before* the write, so a payment is
never punished by its own fingerprint — only by earlier fraud.)

---

## 12. Why the AI "fraud probability" looks like 0% or 100%

You'll notice the raw **fraud probability** is almost always **0.0 % or 100 %**,
rarely in between. **This is expected, not a bug.**

**Reason:** our synthetic features separate fraud from legit *very* sharply (e.g.
device trust ≈ 25 for fraud vs ≈ 82 for legit; VPN 65 % vs 4 %). With ~20 such
clean features, LightGBM can tell the classes apart almost perfectly, so it becomes
**very confident** and outputs probabilities near 0 or 1. Real bank data overlaps
much more, so probabilities would spread out. (The model's tuned threshold is
0.9998, which only happens when it's this confident.)

**Why it doesn't matter for the decision:** the number that drives the decision is
the **Risk Score**, and that is **not** binary — it blends 6 checks (including the
continuous anomaly score), so it spreads nicely: e.g. **19 → 39 → 76 → 89** across
normal → borderline → risky → fraud payments.

**If you want realistic-looking probabilities:** regenerate the synthetic data with
*overlapping* distributions and retrain (~10 min). Probability calibration alone
won't help, because the classes really are separable.

---

## 13. Risk Score vs Customer Trust

These are **two different numbers** — a common point of confusion:

| | **Risk Score** | **Customer Trust** |
|---|---|---|
| Belongs to | **one payment** | **the customer** |
| Range | 0–100, **higher = riskier** | 0–100, **higher = better** |
| Built from | the 6 checks | history of the customer's payments |
| Used for | the decision on this payment | it's *one input* into the Risk Score |

So Customer Trust is a *long-term reputation* that feeds into each payment's
short-term Risk Score.

---

## 14. The "blockchain" — the honest truth

**PayGuard does NOT use real blockchain, and no blockchain framework.** Being
honest about this matters:

- What we call "blockchain" is a **reputation ledger** — a table of receiver/device
  risk scores, stored in **SQLite**.
- The "blockchain log" is just a list of **SHA-256 fingerprints** (hashes) of
  blocked payments — for display.
- There is **no Hyperledger Fabric, no distributed ledger, no consensus, no smart
  contracts, no mining, no network of nodes.**

It's a **simulation** of what a permissioned blockchain would store (shared fraud
intelligence), which is why we call it a "ledger" rather than claiming it's a real
chain.

**Could we use real blockchain?** Yes — the PRD names **Hyperledger Fabric**. But
it's heavy: it needs Docker, a multi-container Fabric network, a channel, and
smart-contract "chaincode." For a demo, the SQLite ledger gives the *same behaviour*
judges care about (shared fraud memory + the feedback loop) without the fragility.
Real Hyperledger is the **production roadmap**. (A lightweight middle option — a
real hash-*chained*, tamper-evident ledger in SQLite — is also possible.)

---

## 15. API reference

Base URL: `http://localhost:8000` · Interactive docs: `/docs`.

| Method | Path | What it does |
|---|---|---|
| POST | `/score` | Score one payment → Risk Score + decision + breakdown + explanation. |
| POST | `/score/batch` | Score up to 100 payments. |
| GET | `/feed` | Recent scored payments (powers the dashboard live feed). |
| GET | `/reputation` | The fraud ledger snapshot. |
| GET | `/reputation/{id}?kind=beneficiary` | Look up one receiver/device. |
| GET | `/governance/reports` | Recent compliance (STR) reports. |
| GET | `/blockchain/log` | Recent fraud fingerprints. |
| GET | `/trust/{customer_id}` | A customer's trust history. |
| GET | `/stats`, `/health` | Session totals and service status. |
| (static) | `/pay/` | The IOB Pay simulator (React app). |

**Example — score a payment:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"amount":85000,"hour":3,"customer_id":"CUST-001",
       "beneficiary_id":"BENF-MULE-001","device_hash":"DEV-9f3a",
       "new_device":true,"vpn_detected":true,"failed_otp_count":2,
       "device_trust_score":20,"geo_distance_km":2500}'
```
Key response fields: `tti` (Risk Score), `tti_breakdown` (the 6 checks),
`decision`, `fraud_probability`, `anomaly_score`, `trust_score` (Customer Trust),
`reasons`, `shap_explanations`, `governance_report`, `response_time_ms`.

---

## 16. The dashboard (analyst console)

Run with `streamlit run dashboard/app.py`. **The dashboard does NOT create
payments** — payments are made in the IOB Pay app (§17). The dashboard is a pure
**investigation console**: it lists every scored payment (live + all past ones from
the database) and lets an analyst **select any one** to see the full story. By
default it talks to the API at `http://localhost:8000` (override with the
`PAYGUARD_API` environment variable).

### Sidebar
- **Engine status** — green when the API is online.
- **"Open IOB Pay app"** button — to go make/simulate payments.
- **Refresh data** — pull the newest payments (there is **no auto-refresh**).
- **Filter by result** — All / Approved / Risky / Blocked.

### Main area
- **Stats row** — totals: Payments / Approved / Risky / Blocked.
- **Transactions** — a dropdown to **select any payment** (newest first) plus a
  colour-coded table of recent payments.
- **🔎 Investigation** (for the selected payment) — the full picture: the
  **decision** and **TTI gauge**, **Customer Trust** (and what moved it), the
  plain-language **reasons**, the **SHAP** explanation, the **TTI breakdown** bar
  chart (the 6 checks and each one's contribution), and the **compliance report
  (STR)** with **download** buttons. This works for **any past transaction too**,
  because full detail is stored in the database.
- **Analytics** — an outcomes pie and a **TTI-over-time** chart.
- **Fraud reputation ledger** — the shared memory of known-bad receivers/devices.

Every chart is a **plot of numbers the engine produced** — the *models* make the
numbers; the *charts* display them.

---

## 17. The IOB Pay simulator

A mock payment app (React, single file) served by the API at
**http://localhost:8000/pay/**.

**Important: it is a SIMULATION — no real money moves.** No UPI/card rails, no bank
account is debited. It sends the payment *details* to PayGuard's `/score` and shows
the decision: **Approved** (success), **Risky** (asks for a simulated OTP), or
**Blocked** (with reasons).

This is where **all transactions originate**. Two ways to create them:
- **Single payment** — pick an amount, a receiver (trusted / new / a flagged one),
  and a device-session profile (Safe / Suspicious / Compromised), then Pay.
- **⚡ Simulate 10 random payments** — one click fires a batch of mixed
  legit/fraud payments through the engine.

Every payment is scored, stored, and then appears in the analyst dashboard (click
**Refresh** there).

> Note: the app loads React from a CDN, so it needs **internet on first load**.

---

## 18. How to run it

```powershell
# 1) Install dependencies
cd d:\demo\fraud-demo
pip install -r requirements.txt

# 2) (only if the .pkl models are missing) regenerate data + retrain
python data/generate_synthetic.py
python models/train_pipeline.py

# 3) Terminal 1 — backend (also serves the IOB Pay app)
cd d:\demo\fraud-demo\api
uvicorn main:app --reload --port 8000       # docs: http://localhost:8000/docs

# 4) Terminal 2 — analyst dashboard
cd d:\demo\fraud-demo
streamlit run dashboard/app.py              # http://localhost:8501
```
Start the backend **first**. The models are already trained (`.pkl` files present),
so step 2 is optional. The SQLite database is created automatically.

---

## 19. Demo script

1. Open **IOB Pay** (http://localhost:8000/pay) and click **⚡ Simulate 10 random
   payments** to populate the system.
2. Make a **single payment** to a trusted receiver → **Approved**; then to
   **"Unknown A/C ••9021"** (a flagged mule) even on a Safe profile → **Blocked**.
3. Open the **dashboard** and click **Refresh data** → all those payments appear.
4. **Select a Blocked payment** → the **Investigation** panel shows the decision,
   TTI gauge, TTI breakdown, **SHAP** explanation, reasons, and the **STR** report
   (download it). Do the same for an Approved one to contrast.
5. Use the **Filter by result** dropdown to jump straight to Blocked payments.
6. Show the **fraud reputation ledger** — the shared memory that caught the mule.
7. **Restart the API** → refresh the dashboard → all history is still there (SQLite).

---

## 20. Roadmap

To make it production-grade (PRD §13/§15):
- **React** analyst console (replacing Streamlit).
- **PostgreSQL** + **Redis** (replacing SQLite / in-memory caches).
- **Docker** for every service; **Grafana** monitoring.
- **Real Hyperledger Fabric** for the shared ledger.
- **Graph Neural Networks** for mule-network discovery; **Federated Learning**;
  cross-bank consortium; real-time threat feeds; voice/deepfake risk.

---

## 21. FAQ

**Q: What database do we use?** SQLite (`data/payguard.db`), via Python's built-in
`sqlite3`. It persists the ledger, customer trust, and transactions across
restarts. PostgreSQL is the production plan.

**Q: Is it real blockchain?** No — it's a SQLite-backed reputation ledger that
*simulates* the shared fraud intelligence a blockchain would hold. See §14.

**Q: Why is fraud probability 0% or 100%?** The model is very confident because the
synthetic data separates the classes cleanly. The **Risk Score** (the number that
decides) is well spread. See §12.

**Q: Are Risk Score and Customer Trust the same?** No — see §13. Risk Score is per
payment (higher = riskier); Customer Trust is per customer (higher = better).

**Q: Is every payment sent to MFA?** No. Only the middle band (Risk 35–70). Safe
payments are approved; clear fraud is blocked.

**Q: Why LightGBM, not deep learning?** On tabular data it's faster, more accurate,
and far more explainable — and explainability is required here.

**Q: Why two models?** LightGBM catches **known** fraud; Isolation Forest catches
**new** fraud. Together they reduce both misses and false alarms.

**Q: Is any personal data stored on the "ledger"?** No — only opaque IDs and a risk
score.

---

*PayGuard · Risk Score (6-check blend) · LightGBM + Isolation Forest + SHAP +
Reputation Ledger + Dynamic Trust Engine · FastAPI + SQLite + Streamlit + React.*

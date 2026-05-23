"""
Continuous Trust Intelligence — Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""
import time, random, requests, json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Trust Intelligence Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.main{background:#0a0f1e}
.block-container{padding:1rem 1.5rem}
div[data-testid="metric-container"]{background:linear-gradient(135deg,#131b2e,#1a2440);border-radius:12px;padding:14px 18px;border:1px solid #1e3050}
.stButton>button{width:100%;border-radius:10px;font-weight:600;transition:all .2s}
.risk-high{color:#ff4757;font-weight:700} .risk-med{color:#ffa502;font-weight:700} .risk-low{color:#2ed573;font-weight:700}
.trust-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-weight:600;font-size:0.85em}
</style>""", unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []
if "chain_log" not in st.session_state: st.session_state.chain_log = []
if "trust_timeline" not in st.session_state: st.session_state.trust_timeline = []
if "customer_id" not in st.session_state: st.session_state.customer_id = "CUST-001"

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🛡️ Trust Intelligence")
    st.markdown("*Continuous Fraud Prevention*")
    st.divider()
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.ok:
            h = r.json()
            st.success(f"✓ API Online — {h.get('mode','ML')}")
            st.caption(f"Features: {h.get('features',0)} | Customers: {h.get('customers_tracked',0)}")
        else: st.error("API error")
    except: st.warning("⚠️ API offline — start API first\n```\ncd api\nuvicorn main:app --reload\n```")

    st.divider()
    st.markdown("### 👤 Customer")
    cust_id = st.text_input("Customer ID", st.session_state.customer_id)
    st.session_state.customer_id = cust_id

    st.divider()
    st.markdown("### ⚡ Quick Scenarios")
    c1, c2, c3 = st.columns(3)
    preset = None
    if c1.button("✅ Normal"): preset = "normal"
    if c2.button("⚠️ Risky"): preset = "risky"
    if c3.button("🚫 Fraud"): preset = "fraud"

    presets = {
        "normal": dict(amount=3500, hour=14, avg=20000, new_dev=False, vpn=False, emu=False,
                       root=False, paste=False, otp=0, benef=False, intl=False, geo=5,
                       session=45, typing=0.1, shared=1, benef_risk=10, age=365, dev_trust=85),
        "risky":  dict(amount=85000, hour=2, avg=20000, new_dev=True, vpn=True, emu=False,
                       root=False, paste=True, otp=1, benef=True, intl=False, geo=200,
                       session=8, typing=0.6, shared=2, benef_risk=45, age=30, dev_trust=40),
        "fraud":  dict(amount=450000, hour=3, avg=15000, new_dev=True, vpn=True, emu=True,
                       root=True, paste=True, otp=3, benef=True, intl=True, geo=2500,
                       session=4, typing=0.9, shared=5, benef_risk=80, age=3, dev_trust=15),
    }
    if preset and preset in presets:
        for k, v in presets[preset].items(): st.session_state[k] = v

    st.divider()
    st.markdown("### 💳 Transaction")
    amount = st.slider("Amount (₹)", 100, 500000, st.session_state.get("amount", 5000), step=100, format="₹%d")
    hour = st.slider("Hour", 0, 23, st.session_state.get("hour", 14))
    avg_monthly = st.number_input("User Avg Monthly (₹)", 1000, 500000, st.session_state.get("avg", 20000), step=500)

    st.markdown("### 📱 Device Signals")
    new_device = st.checkbox("New Device", st.session_state.get("new_dev", False))
    rooted = st.checkbox("Rooted/Jailbroken", st.session_state.get("root", False))
    emulator = st.checkbox("Emulator", st.session_state.get("emu", False))
    dev_trust = st.slider("Device Trust", 0, 100, st.session_state.get("dev_trust", 80))

    st.markdown("### 🌍 Geo Signals")
    vpn = st.checkbox("VPN Detected", st.session_state.get("vpn", False))
    intl = st.checkbox("International", st.session_state.get("intl", False))
    geo_dist = st.number_input("Geo Distance (km)", 0, 15000, st.session_state.get("geo", 5))

    st.markdown("### ⚡ Session Signals")
    session_dur = st.slider("Session Duration (s)", 1, 600, st.session_state.get("session", 45))
    paste = st.checkbox("Paste Detected", st.session_state.get("paste", False))
    otp_fails = st.slider("Failed OTPs", 0, 5, st.session_state.get("otp", 0))
    benef_new = st.checkbox("New Beneficiary", st.session_state.get("benef", False))
    typing_anom = st.slider("Typing Anomaly", 0.0, 1.0, st.session_state.get("typing", 0.1))

    st.markdown("### 🔗 Relationship")
    shared_dev = st.slider("Shared Device Accounts", 1, 10, st.session_state.get("shared", 1))
    benef_risk = st.slider("Beneficiary Risk", 0, 100, st.session_state.get("benef_risk", 10))
    acct_age = st.number_input("Account Age (days)", 1, 3650, st.session_state.get("age", 365))

    st.divider()
    analyze_btn = st.button("🔍 Analyze Transaction", type="primary")
    simulate_btn = st.button("⚡ Simulate 10 Txns")

# ── API Call ──
def call_api(payload):
    try:
        r = requests.post(f"{API_URL}/score", json=payload, timeout=10)
        return r.json()
    except:
        return fallback_score(payload)

def fallback_score(p):
    score = 0; reasons = []
    avg = p.get("avg_txn_amount", 5000) or 5000
    r = p["amount"] / avg
    if r > 10: score += 40; reasons.append(f"Amount {r:.1f}x above avg")
    elif r > 5: score += 25; reasons.append(f"Amount {r:.1f}x above avg")
    if p["hour"] < 5: score += 20; reasons.append("Late night")
    if p.get("new_device"): score += 15; reasons.append("New device")
    if p.get("vpn_detected"): score += 10; reasons.append("VPN detected")
    if p.get("emulator_detected"): score += 12; reasons.append("Emulator")
    if p.get("paste_detected"): score += 10; reasons.append("Paste detected")
    if (p.get("failed_otp_count") or 0) >= 2: score += 15; reasons.append(f"Failed OTP x{p['failed_otp_count']}")
    if p.get("beneficiary_added_recently"): score += 12; reasons.append("New beneficiary")
    score = min(score, 100)
    dec = "APPROVE" if score < 35 else ("STEP_UP_MFA" if score < 70 else "BLOCK")
    return {"transaction_id": f"L-{random.randint(1000,9999)}", "risk_score": score,
            "decision": dec, "fraud_probability": score/100, "anomaly_score": score/100,
            "trust_score": 85-score*0.5, "trust_delta": -score*0.3,
            "decision_confidence": 0.7, "reasons": reasons,
            "shap_explanations": [], "trust_factors": [],
            "response_time_ms": 1.5, "blockchain_hash": None,
            "governance_report": {}, "timestamp": datetime.utcnow().isoformat()}

def add_history(res, amt):
    merchants = ["Amazon IN","Flipkart","Swiggy","PhonePe","Paytm","Zomato","HDFC","SBI","ICICI","Google Pay"]
    st.session_state.history.insert(0, {"TX ID": res["transaction_id"], "Amount": f"₹{amt:,.0f}",
        "Decision": res["decision"], "Risk": res["risk_score"], "Trust": res.get("trust_score", "—"),
        "Merchant": random.choice(merchants), "Time": datetime.now().strftime("%H:%M:%S")})
    if "trust_score" in res:
        st.session_state.trust_timeline.append({"trust": res["trust_score"], "time": datetime.now().strftime("%H:%M:%S"),
            "txn": res["transaction_id"]})
    if res.get("blockchain_hash"):
        st.session_state.chain_log.insert(0, {"Hash": res["blockchain_hash"], "Score": res["risk_score"],
            "Amount": f"₹{amt:,.0f}", "Time": res["timestamp"][:19].replace("T"," ")})
    if len(st.session_state.history) > 50: st.session_state.history.pop()

# ── Main Dashboard ──
st.markdown("# 🛡️ Continuous Trust Intelligence Dashboard")
st.markdown("*Real-time fraud prevention · LightGBM + Isolation Forest + SHAP + Dynamic Trust Engine*")
st.divider()

result = None
if analyze_btn:
    payload = {"amount": amount, "hour": hour, "customer_id": cust_id,
        "avg_txn_amount": avg_monthly, "txn_frequency": 1.5, "txn_count_1h": 1, "txn_count_24h": 3,
        "new_device": new_device, "rooted_device": rooted, "emulator_detected": emulator,
        "device_trust_score": dev_trust, "geo_distance_km": geo_dist,
        "vpn_detected": vpn, "is_international": intl,
        "session_duration_sec": session_dur, "paste_detected": paste,
        "failed_otp_count": otp_fails, "beneficiary_added_recently": benef_new,
        "typing_speed_anomaly": typing_anom, "shared_device_accounts": shared_dev,
        "beneficiary_risk_score": benef_risk, "account_age_days": acct_age}
    with st.spinner("Analyzing..."): time.sleep(0.5); result = call_api(payload)
    if result: add_history(result, amount)

if simulate_btn:
    prog = st.progress(0, "Simulating...")
    for i in range(10):
        is_fraud = random.random() > 0.7
        sa = random.choice([800, 2500, 5000, 15000, 85000, 200000, 450000] if is_fraud else [500, 1200, 3000, 5000, 8000])
        payload = {"amount": sa, "hour": random.randint(0,23), "customer_id": cust_id,
            "avg_txn_amount": 20000, "new_device": random.random()>0.5 if is_fraud else False,
            "vpn_detected": random.random()>0.4 if is_fraud else False,
            "emulator_detected": random.random()>0.7 if is_fraud else False,
            "rooted_device": random.random()>0.6 if is_fraud else False,
            "paste_detected": random.random()>0.4 if is_fraud else False,
            "failed_otp_count": random.choice([0,1,2,3]) if is_fraud else 0,
            "beneficiary_added_recently": random.random()>0.5 if is_fraud else False,
            "is_international": random.random()>0.6 if is_fraud else False,
            "geo_distance_km": random.choice([500,1000,3000]) if is_fraud else random.randint(1,50),
            "device_trust_score": random.randint(10,40) if is_fraud else random.randint(70,95),
            "session_duration_sec": random.choice([3,5,8]) if is_fraud else random.randint(20,120),
            "typing_speed_anomaly": round(random.uniform(0.5,0.9),2) if is_fraud else round(random.uniform(0.05,0.2),2),
            "shared_device_accounts": random.choice([3,5,8]) if is_fraud else 1,
            "beneficiary_risk_score": random.randint(50,90) if is_fraud else random.randint(5,20),
            "account_age_days": random.choice([1,3,7]) if is_fraud else random.randint(90,730)}
        r = call_api(payload)
        if r: add_history(r, sa)
        prog.progress((i+1)/10, f"Transaction {i+1}/10...")
        time.sleep(0.15)
    prog.empty(); st.success("✓ 10 transactions simulated!")

# ── Result Panel ──
if result:
    dec = result["decision"]
    score = result["risk_score"]
    colors = {"APPROVE": "#2ed573", "STEP_UP_MFA": "#ffa502", "BLOCK": "#ff4757"}
    icons = {"APPROVE": "✅", "STEP_UP_MFA": "⚠️", "BLOCK": "🚫"}
    color = colors.get(dec, "#999")

    c1, c2, c3, c4 = st.columns([1.2, 2, 1.5, 1.5])
    with c1:
        st.markdown(f"### {icons.get(dec,'')} {dec}")
        st.metric("Risk Score", f"{score}/100")
        st.metric("Fraud Prob", f"{result['fraud_probability']*100:.1f}%")
        st.metric("Response", f"{result['response_time_ms']} ms")
    with c2:
        fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=score,
            delta={"reference": 50}, title={"text": "Risk Score", "font": {"size": 16}},
            gauge={"axis": {"range": [0,100]}, "bar": {"color": color},
                "steps": [{"range":[0,35],"color":"#1a3a1a"},{"range":[35,70],"color":"#3a3a1a"},
                          {"range":[70,100],"color":"#3a1a1a"}],
                "threshold": {"line":{"color":color,"width":4},"value":score}}))
        fig.update_layout(height=220, margin=dict(l=20,r=20,t=40,b=10),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        trust = result.get("trust_score", 85)
        delta = result.get("trust_delta", 0)
        conf = result.get("decision_confidence", 0.5)
        st.markdown("#### 🔐 Trust Engine")
        st.metric("Trust Score", f"{trust:.0f}/100", delta=f"{delta:+.1f}")
        st.metric("Confidence", f"{conf*100:.0f}%")
        # Trust factors
        factors = result.get("trust_factors", [])
        if factors:
            for f in factors[:5]:
                icon = "🔴" if f.get("impact",0) < 0 else "🟢"
                st.caption(f"{icon} {f.get('factor','')} ({f.get('impact',0):+.1f})")
    with c4:
        st.markdown("#### 🧠 SHAP Explainability")
        shap_data = result.get("shap_explanations", [])
        if shap_data:
            for s in shap_data[:5]:
                icon = "🔴" if s.get("impact","") == "increases_risk" else "🟢"
                st.caption(f"{icon} **{s['feature']}**: {s['shap_value']:.3f}")
        st.markdown("#### 📋 Risk Reasons")
        for reason in result.get("reasons", [])[:5]:
            st.caption(f"• {reason}")

    # Governance report
    gov = result.get("governance_report", {})
    if gov:
        with st.expander("📜 Governance / STR Report"):
            st.json(gov)
    st.divider()

# ── Analytics ──
history = st.session_state.history
if history:
    df_hist = pd.DataFrame(history)
    total = len(df_hist)
    blocked = (df_hist["Decision"]=="BLOCK").sum()
    mfa = (df_hist["Decision"]=="STEP_UP_MFA").sum()
    approved = (df_hist["Decision"]=="APPROVE").sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Analyzed", total)
    m2.metric("✅ Approved", approved, delta=f"{approved/total*100:.0f}%")
    m3.metric("⚠️ MFA", mfa, delta=f"{mfa/total*100:.0f}%")
    m4.metric("🚫 Blocked", blocked, delta=f"-{blocked/total*100:.0f}%")

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("#### Decision Distribution")
        dec_counts = df_hist["Decision"].value_counts()
        fig_pie = px.pie(values=dec_counts.values, names=dec_counts.index,
            color=dec_counts.index, color_discrete_map={"APPROVE":"#2ed573","STEP_UP_MFA":"#ffa502","BLOCK":"#ff4757"}, hole=0.45)
        fig_pie.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc", legend=dict(orientation="h",y=-0.1))
        st.plotly_chart(fig_pie, use_container_width=True)
    with ch2:
        st.markdown("#### Risk Score Timeline")
        df_plot = df_hist.copy(); df_plot["Index"] = range(len(df_plot)); df_plot = df_plot.iloc[::-1]
        fig_sc = px.scatter(df_plot, x="Index", y="Risk", color="Decision",
            color_discrete_map={"APPROVE":"#2ed573","STEP_UP_MFA":"#ffa502","BLOCK":"#ff4757"}, size=[8]*len(df_plot))
        fig_sc.add_hline(y=35, line_dash="dash", line_color="#2ed573", opacity=0.5)
        fig_sc.add_hline(y=70, line_dash="dash", line_color="#ff4757", opacity=0.5)
        fig_sc.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc", legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig_sc, use_container_width=True)

    # Trust timeline
    if st.session_state.trust_timeline:
        st.markdown("#### 🔐 Trust Score Evolution")
        df_trust = pd.DataFrame(st.session_state.trust_timeline)
        fig_trust = px.line(df_trust, x="time", y="trust", markers=True,
            labels={"trust": "Trust Score", "time": "Transaction"})
        fig_trust.add_hline(y=70, line_dash="dash", line_color="#2ed573", opacity=0.5, annotation_text="Trusted")
        fig_trust.add_hline(y=40, line_dash="dash", line_color="#ff4757", opacity=0.5, annotation_text="Untrusted")
        fig_trust.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        fig_trust.update_traces(line_color="#7c3aed")
        st.plotly_chart(fig_trust, use_container_width=True)

    st.markdown("#### 📋 Transaction Feed")
    def color_dec(val):
        c = {"APPROVE":"#2ed573","STEP_UP_MFA":"#ffa502","BLOCK":"#ff4757"}.get(val,"")
        return f"color:{c};font-weight:bold"
    styled = df_hist.head(20).style.applymap(color_dec, subset=["Decision"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Blockchain Log ──
if st.session_state.chain_log:
    st.divider()
    st.markdown("#### ⛓️ Blockchain Fraud Pattern Log")
    st.caption("Fraud fingerprints stored on Hyperledger Fabric (simulated)")
    st.dataframe(pd.DataFrame(st.session_state.chain_log), use_container_width=True, hide_index=True)
else:
    st.info("No fraud patterns yet. Analyze a transaction to begin.")

st.divider()
st.caption("Continuous Trust Intelligence Framework · LightGBM + IsoForest + SHAP + Dynamic Trust Engine · FastAPI + Streamlit")

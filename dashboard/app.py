"""
PaymentGuardian — Fraud Analyst Console (investigation dashboard)
==========================================================
This dashboard does NOT create payments. Payments are made in the IOB Pay app
(http://localhost:8000/pay). This console lists every scored payment (live + all
past ones from the database), and lets an analyst SELECT any payment to see the
full investigation: decision, TTI breakdown, SHAP explanation, reasons, customer
trust, and the compliance (STR) report.

Run: streamlit run dashboard/app.py
"""
import os, json, requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_URL = os.environ.get("PAYMENTGUARDIAN_API", "http://localhost:8000")

st.set_page_config(page_title="PaymentGuardian — Analyst Console", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.main{background:#0e1420}
.block-container{padding:1.2rem 2rem}
div[data-testid="stMetric"]{background:#161d2b;border-radius:12px;padding:14px 18px;border:1px solid #24304a}
.stButton>button{width:100%;border-radius:10px;font-weight:600}
.cap{color:#8b98ad;font-size:0.86rem}
</style>""", unsafe_allow_html=True)

FRIENDLY = {"APPROVE": "Approved", "STEP_UP_MFA": "Risky", "BLOCK": "Blocked"}
COLORS   = {"Approved": "#22c55e", "Risky": "#f59e0b", "Blocked": "#ef4444"}
ICONS    = {"Approved": "✅", "Risky": "⚠️", "Blocked": "⛔"}
SUBTITLE = {"Approved": "Payment allowed", "Risky": "Needs extra verification (OTP)", "Blocked": "Payment stopped"}
label = lambda d: FRIENDLY.get(d, d)

def api_get(path, default=None):
    try:
        return requests.get(f"{API_URL}{path}", timeout=2).json()
    except Exception:
        return default

def str_report_text(gov):
    lines = ["SUSPICIOUS TRANSACTION REPORT (STR)", "=" * 52,
        f"STR ID           : {gov.get('str_id','—')}",
        f"Generated At     : {gov.get('generated_at','—')}",
        f"Regulatory Basis : {gov.get('regulatory_basis','—')}", "",
        "TRANSACTION", "-" * 52,
        f"Transaction ID   : {gov.get('transaction_id','—')}",
        f"Customer ID      : {gov.get('customer_id','—')}",
        f"Amount           : INR {gov.get('amount','—')}",
        f"TTI (Risk Score) : {gov.get('risk_score','—')}/100",
        f"Result           : {label(gov.get('decision',''))}",
        f"Customer Trust   : {gov.get('trust_score','—')}/100", "",
        "RISK INDICATORS", "-" * 52]
    for i, ind in enumerate(gov.get("risk_indicators", []), 1):
        lines.append(f"  {i}. {ind}")
    lines += ["", "ACTION REQUIRED", "-" * 52, f"  {gov.get('action_required','—')}", ""]
    return "\n".join(lines)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🛡️ PaymentGuardian")
    st.caption("Fraud analyst console")
    st.divider()
    h = api_get("/health")
    if h:
        st.success(f"Engine online — {h.get('mode','ML')}")
        st.caption(f"{h.get('features',0)} signals · {h.get('customers_tracked',0)} customers")
    else:
        st.warning("Engine offline — start the API:\n```\ncd api\nuvicorn main:app --reload\n```")
    st.divider()
    st.markdown("### Make payments")
    st.caption("Payments are created in the IOB Pay app. New payments appear here after you refresh.")
    st.link_button("💳 Open IOB Pay app", "http://localhost:8000/pay/")
    st.divider()
    refresh = st.button("🔄 Refresh data", type="primary")
    dec_filter = st.selectbox("Filter by result", ["All", "Approved", "Risky", "Blocked"])

# ── Header ──
st.markdown("# 🛡️ PaymentGuardian — Fraud Analyst Console")
st.markdown("<span class='cap'>Every payment scored by the engine (live + history). "
            "Select any payment below to investigate why it was Approved, flagged Risky, or Blocked.</span>",
            unsafe_allow_html=True)
st.divider()

# ── Load transactions from the database ──
data = api_get("/transactions?limit=300", {"transactions": []})
txns = data.get("transactions", [])

if not txns:
    st.info("No payments yet. Open the **IOB Pay app** (http://localhost:8000/pay), make or simulate "
            "some payments, then click **🔄 Refresh data**.")
    st.stop()

df = pd.DataFrame(txns)
df["Result"] = df["decision"].map(label)
df["Time"] = df["timestamp"].astype(str).str[11:19]
if dec_filter != "All":
    df = df[df["Result"] == dec_filter]

# ── Stats ──
total = len(df)
approved = int((df["Result"] == "Approved").sum())
risky = int((df["Result"] == "Risky").sum())
blocked = int((df["Result"] == "Blocked").sum())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Payments", total)
m2.metric("✅ Approved", approved)
m3.metric("⚠️ Risky", risky)
m4.metric("⛔ Blocked", blocked)
st.divider()

# ── Transaction picker + list ──
st.markdown("### Transactions")
st.caption("Newest first. Pick one to investigate it in detail below.")

records = {t["transaction_id"]: t for t in df.to_dict("records")}
def opt_label(tid):
    t = records[tid]
    return f"{t['Time']} · ₹{t.get('amount',0):,.0f} · {t['Result']} · TTI {t.get('tti','—')} · {t.get('beneficiary_name') or t.get('beneficiary_id') or '—'} · {tid}"

choice_id = st.selectbox("Select a transaction to investigate",
                         options=list(records.keys()), format_func=opt_label, index=0)

# compact table of all (colour by result)
show = df[["Time", "channel", "beneficiary_name", "amount", "tti", "Result", "top_reason"]].copy()
show["amount"] = show["amount"].map(lambda a: f"₹{a:,.0f}" if pd.notna(a) else "—")
show.columns = ["Time", "Channel", "Beneficiary", "Amount", "TTI", "Result", "Top reason"]
_s = show.head(25).style
_m = _s.map if hasattr(_s, "map") else _s.applymap
st.dataframe(_m(lambda v: f"color:{COLORS.get(v,'')};font-weight:bold" if v in COLORS else "",
                subset=["Result"]), use_container_width=True, hide_index=True)
st.divider()

# ── Investigation of the selected transaction ──
rec = api_get(f"/transaction/{choice_id}")
detail = (rec or {}).get("detail")

st.markdown(f"### 🔎 Investigation — `{choice_id}`")
if not detail:
    st.warning("No stored detail for this transaction (it was scored before detail-capture was added). "
               "Summary only. New payments will have full detail.")
    st.json(records[choice_id])
else:
    dec = label(detail["decision"])
    tti = detail.get("tti", 0)
    color = COLORS.get(dec, "#999")
    raw = detail.get("raw", {})

    c1, c2, c3 = st.columns([1.3, 2, 2])
    with c1:
        st.markdown(f"### {ICONS.get(dec,'')} {dec}")
        st.caption(SUBTITLE.get(dec, ""))
        st.metric("TTI (Trust Index)", f"{tti}/100")
        st.caption(f"₹{detail.get('amount',0):,.0f} → {detail.get('beneficiary_name') or detail.get('beneficiary_id') or '—'}")
        st.caption(f"Customer {detail.get('customer_id','—')} · via {detail.get('channel','—')}")
        st.caption(f"Fraud probability {detail['fraud_probability']*100:.1f}% · {detail['response_time_ms']} ms")
    with c2:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=tti,
            title={"text": "Transaction Trust Index — TTI (higher = riskier)", "font": {"size": 14}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                "steps": [{"range": [0, 35], "color": "#173a26"}, {"range": [35, 70], "color": "#3a3117"},
                          {"range": [70, 100], "color": "#3a1a1a"}],
                "threshold": {"line": {"color": color, "width": 4}, "value": tti}}))
        fig.update_layout(height=210, margin=dict(l=20, r=20, t=40, b=6),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Under 35 → Approved · 35–70 → Risky (verify) · 70+ → Blocked")
    with c3:
        st.markdown("#### Customer trust")
        st.caption("The customer's reputation, built over their history (higher = better).")
        st.metric("Customer Trust", f"{detail.get('trust_score',85):.0f}/100",
                  delta=f"{detail.get('trust_delta',0):+.1f}")
        for f in detail.get("trust_factors", [])[:3]:
            icon = "🔴" if f.get("impact", 0) < 0 else "🟢"
            st.caption(f"{icon} {f.get('factor','')} ({f.get('impact',0):+.1f})")

    if raw:
        st.markdown("#### Transaction details")
        st.caption("Everything the engine received about this payment — account, device, location and session signals.")
        FIELD_GROUPS = [
            ("Account & behaviour", [
                ("account_age_days", "Account age", "{} days"),
                ("avg_txn_amount", "Avg. spend", "₹{:,.0f}"),
                ("txn_frequency", "Txn frequency", "{:.1f}/day"),
                ("txn_count_1h", "Payments in last hour", "{}"),
                ("txn_count_24h", "Payments in last 24h", "{}"),
                ("hour", "Hour of day", "{:02.0f}:00"),
            ]),
            ("Device", [
                ("device_trust_score", "Device trust", "{:.0f}/100"),
                ("new_device", "New / unknown device", None),
                ("rooted_device", "Rooted / jailbroken", None),
                ("emulator_detected", "Emulator detected", None),
            ]),
            ("Location", [
                ("geo_distance_km", "Distance from last txn", "{:.0f} km"),
                ("vpn_detected", "VPN / proxy", None),
                ("is_international", "International", None),
            ]),
            ("Session", [
                ("session_duration_sec", "Time on page", "{:.0f}s"),
                ("paste_detected", "Account no. pasted", None),
                ("failed_otp_count", "Failed OTP attempts", "{}"),
                ("typing_speed_anomaly", "Typing anomaly", "{:.2f}"),
            ]),
            ("Relationship", [
                ("beneficiary_added_recently", "New beneficiary", None),
                ("beneficiary_risk_score", "Beneficiary risk", "{:.0f}/100"),
                ("shared_device_accounts", "Accounts sharing device", "{}"),
            ]),
        ]
        d_cols = st.columns(len(FIELD_GROUPS))
        for d_col, (group_name, fields) in zip(d_cols, FIELD_GROUPS):
            with d_col:
                st.markdown(f"**{group_name}**")
                for key, lbl, fmt in fields:
                    val = raw.get(key)
                    if val is None:
                        continue
                    if fmt is None:
                        icon = "🔴" if val else "🟢"
                        st.caption(f"{icon} {lbl}: {'Yes' if val else 'No'}")
                    else:
                        try:
                            st.caption(f"{lbl}: {fmt.format(val)}")
                        except (ValueError, TypeError):
                            st.caption(f"{lbl}: {val}")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("#### Why this decision")
        st.caption("Plain-language risk signals for this payment.")
        for reason in detail.get("reasons", [])[:8]:
            st.markdown(f"- {reason}")
    with r2:
        st.markdown("#### What the AI model weighed (SHAP)")
        st.caption("Features that pushed the model's fraud probability up (🔴) or down (🟢).")
        shap_data = detail.get("shap_explanations", [])
        if shap_data:
            for s in shap_data[:8]:
                icon = "🔴" if s.get("impact", "") == "increases_risk" else "🟢"
                fval = raw.get(s["feature"])
                fval_txt = f" *(value: {fval})*" if fval is not None else ""
                st.markdown(f"{icon} **{s['feature']}**{fval_txt} — {s['shap_value']:.3f}")
        else:
            st.caption("No SHAP output for this transaction.")

    breakdown = detail.get("tti_breakdown", [])
    if breakdown:
        st.markdown("#### TTI breakdown — the 6 checks")
        st.caption("The TTI is a weighted blend of six checks. Each bar = that check's contribution "
                   "(its 0–100 sub-score × its weight).")
        bd = pd.DataFrame(breakdown).iloc[::-1]
        fig_bd = go.Figure(go.Bar(
            y=bd["factor"], x=bd["contribution"], orientation="h",
            marker=dict(color=bd["score"], colorscale="RdYlGn_r", cmin=0, cmax=100,
                        colorbar=dict(title="Risk", thickness=12)),
            text=[f"{s:.0f} × {w} = {c:.1f}" for s, w, c in zip(bd["score"], bd["weight"], bd["contribution"])],
            textposition="auto"))
        fig_bd.update_layout(height=300, margin=dict(l=10, r=10, t=6, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc",
            xaxis_title="Contribution to TTI")
        st.plotly_chart(fig_bd, use_container_width=True)
        for b in breakdown:
            if b.get("note"): st.caption(f"• {b['factor']}: {b['note']}")

    gov = detail.get("governance_report", {})
    if gov:
        with st.expander("Compliance report (STR) — download"):
            st.json(gov)
            d1, d2 = st.columns(2)
            d1.download_button("Download STR (.txt)", str_report_text(gov),
                file_name=f"{gov.get('str_id','STR')}.txt", mime="text/plain")
            d2.download_button("Download STR (.json)", json.dumps(gov, indent=2),
                file_name=f"{gov.get('str_id','STR')}.json", mime="application/json")

st.divider()

# ── Session analytics (across all loaded transactions) ──
st.markdown("### Analytics")
a1, a2 = st.columns(2)
with a1:
    st.markdown("#### Outcomes")
    st.caption("Share of payments by result.")
    vc = df["Result"].value_counts()
    fig_pie = px.pie(values=vc.values, names=vc.index, color=vc.index,
                     color_discrete_map=COLORS, hole=0.45)
    fig_pie.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc", legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig_pie, use_container_width=True)
with a2:
    st.markdown("#### TTI over time")
    st.caption("Each dot is one payment; lines mark the Approve / Risky / Block bands.")
    dfp = df.iloc[::-1].reset_index(drop=True); dfp["#"] = range(len(dfp))
    fig_sc = px.scatter(dfp, x="#", y="tti", color="Result", color_discrete_map=COLORS,
                        hover_data=["transaction_id", "amount"], size=[8] * len(dfp))
    fig_sc.add_hline(y=35, line_dash="dash", line_color="#22c55e", opacity=0.5)
    fig_sc.add_hline(y=70, line_dash="dash", line_color="#ef4444", opacity=0.5)
    fig_sc.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc", legend=dict(orientation="h", y=-0.2),
        yaxis_title="TTI")
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Reputation ledger ──
st.divider()
st.markdown("### Fraud reputation ledger")
st.caption("Shared memory of known-bad receivers and devices. A confirmed fraud raises an entity's risk, "
           "so future payments to it are caught automatically.")
rep = api_get("/reputation")
if rep:
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Entities", rep.get("total_entities", 0))
    g2.metric("Receivers", rep.get("beneficiaries", 0))
    g3.metric("Devices", rep.get("devices", 0))
    g4.metric("Flagged", rep.get("flagged", 0))
    entries = rep.get("entries", [])
    if entries:
        led = pd.DataFrame(entries)[["entity_id", "kind", "risk", "fraud_reports", "note"]]
        led.columns = ["Entity", "Type", "Risk", "Fraud reports", "Note"]
        st.dataframe(led, use_container_width=True, hide_index=True)
else:
    st.info("Ledger unavailable — start the API.")

st.divider()
st.caption("PaymentGuardian · Transaction Trust Index / TTI (6-check blend) · AI model + anomaly detection + explainability + shared fraud memory")

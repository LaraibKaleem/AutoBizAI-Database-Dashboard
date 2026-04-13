# =============================================================================
# MODULE 5 — DASHBOARD
# File: app.py
# Run: streamlit run app.py
#
# Pages:
#   Overview    — KPI cards, alert summary, top alerts
#   Fraud       — Fraud orders, charts, XAI explanations
#   Delivery    — Late risk predictions, charts, order detail
#   Inventory   — Stock levels, reorder alerts, demand forecast
#   All Alerts  — Filterable alert feed, resolve actions
# =============================================================================

# import sys, os
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import execute_query, get_alerts, get_fraud_orders, get_inventory_status, get_predictions, get_weekly_demand, resolve_alert
# from modules.database import execute_query,get_fraud_orders, get_inventory_status,resolve_alert

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoBiz AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Clean font and background */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #f8f9fb; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data]
/* Sidebar labels (AutoBiz AI) */
[data-testid="stSidebar"] * {
    font-size: 16px; 
}  
[data-testid="stSidebar"] * 
    { color: #ffffff !important;}

/* Sidebar buttons */
[data-testid="stSidebar"] button {
    font-size: 16px;
    background-color: #2563eb;
    color: white;
}
/* BUG 3 FIX — keep sidebar collapse/expand toggle always visible */
[data-testid="collapsedControl"] {
    display: flex !important;
    background: #0f1117 !important;
    border-right: 1px solid #1e2130 !important;
    color: #ffffff !important;
}
[data-testid="collapsedControl"] * {
    color: #e0e0e0 !important;
}

/* KPI cards */
.kpi-card {
    background: white;
    border-radius: 10px;
    padding: 18px 20px;
    border-left: 4px solid #4361ee;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
.kpi-card.red   { border-left-color: #ef233c; }
.kpi-card.amber { border-left-color: #f77f00; }
.kpi-card.green { border-left-color: #06d6a0; }
.kpi-card.grey  { border-left-color: #8d99ae; }
.kpi-label { font-size: 12px; color: #6b7280; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #111827; line-height: 1; }
.kpi-sub   { font-size: 12px; color: #9ca3af; margin-top: 4px; }

/* Section headers */
.section-header {
    font-size: 20px; font-weight: 700; color: #374151;
    text-transform: uppercase; letter-spacing: 0.08em;
    border-bottom: 2px solid #e5e7eb; padding-bottom: 6px;
    margin: 20px 0 12px 0;
}

/* Severity badges */
.badge-critical { background:#fef2f2;color:#dc2626;padding:2px 8px;
                  border-radius:4px;font-size:11px;font-weight:700; }
.badge-high     { background:#fff7ed;color:#ea580c;padding:2px 8px;
                  border-radius:4px;font-size:11px;font-weight:700; }
.badge-medium   { background:#fefce8;color:#ca8a04;padding:2px 8px;
                  border-radius:4px;font-size:11px;font-weight:700; }
.badge-low      { background:#f0fdf4;color:#16a34a;padding:2px 8px;
                  border-radius:4px;font-size:11px;font-weight:700; }

/* Alert card */
.alert-card {
    background: white; border-radius: 8px; padding: 14px 16px;
    margin-bottom: 10px; border-left: 3px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.alert-card.critical { border-left-color: #dc2626; }
.alert-card.high     { border-left-color: #ea580c; }
.alert-card.medium   { border-left-color: #ca8a04; }
.alert-meta  { font-size: 11px; color: #9ca3af; margin-top: 6px; }
.alert-msg   { font-size: 13px; color: #374151; margin: 6px 0; }
.alert-rec   { font-size: 12px; color: #6b7280; font-style: italic; }

/* Table styling */
.dataframe { font-size: 13px !important; }
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS — cached, safe
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=120, show_spinner=False)
def load_kpis():
    try:
        r = execute_query("SELECT COUNT(*), SUM(sales) FROM orders")
        total_orders = int(r["rows"][0][0] or 0)
        total_revenue = float(r["rows"][0][1] or 0)

        r2 = execute_query("SELECT COUNT(*) FROM orders WHERE fraud_label=1")
        fraud_count = int(r2["rows"][0][0] or 0)

        r3 = execute_query("SELECT COUNT(*) FROM predictions WHERE prediction_type='delivery' AND label='late'")
        delivery_risk = int(r3["rows"][0][0] or 0)

        r4 = execute_query("SELECT COUNT(*) FROM predictions WHERE prediction_type='inventory' AND label='critical'")
        critical_stock = int(r4["rows"][0][0] or 0)

        r5 = execute_query("SELECT COUNT(*) FROM alerts WHERE is_resolved=0")
        open_alerts = int(r5["rows"][0][0] or 0)

        r6 = execute_query("""
            SELECT SUM(p.predicted_value) FROM predictions p
            INNER JOIN (SELECT product_name, MIN(label) AS ml FROM predictions
                        WHERE prediction_type='demand' GROUP BY product_name) sub
            ON p.product_name=sub.product_name AND p.label=sub.ml
            WHERE p.prediction_type='demand'
        """)
        next_week = int(float(r6["rows"][0][0] or 0))

        return {
            "total_orders": total_orders, "total_revenue": total_revenue,
            "fraud_count": fraud_count, "delivery_risk": delivery_risk,
            "critical_stock": critical_stock, "open_alerts": open_alerts,
            "next_week_demand": next_week,
        }
    except Exception as e:
        return {}


@st.cache_data(ttl=120, show_spinner=False)
def load_alerts_df(category=None, severity=None, resolved=False, limit=500):
    try:
        q  = "SELECT id,created_at,agent,severity,category,product_name,order_id,message,recommendation,is_resolved FROM alerts WHERE 1=1"
        p  = []
        if category:
            q += " AND category=?"; p.append(category)
        if severity:
            q += " AND severity=?"; p.append(severity)
        if not resolved:
            q += " AND is_resolved=0"
        q += " ORDER BY CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END, created_at DESC LIMIT ?"
        p.append(limit)
        r = execute_query(q, p)
        cols = ["id","created_at","agent","severity","category",
                "product_name","order_id","message","recommendation","is_resolved"]
        return pd.DataFrame(r["rows"], columns=cols)
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def load_fraud_df():
    try:
        df = get_fraud_orders(limit=500)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def load_inventory_df():
    try:
        return get_inventory_status()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def load_delivery_df():
    try:
        r = execute_query("""
            SELECT p.order_id, p.predicted_value, a.severity, a.message, a.recommendation,
                   a.product_name
            FROM predictions p
            LEFT JOIN alerts a ON a.order_id=p.order_id AND a.category='delivery'
            WHERE p.prediction_type='delivery' AND p.label='late'
            ORDER BY p.predicted_value DESC LIMIT 200
        """)
        cols = ["order_id","risk_prob","severity","message","recommendation","product_name"]
        df = pd.DataFrame(r["rows"], columns=cols)
        df["risk_prob"] = df["risk_prob"].astype(float)
        df["risk_pct"]  = (df["risk_prob"] * 100).round(1).astype(str) + "%"
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def load_demand_forecast(product_name):
    try:
        r = execute_query("""
            SELECT label, predicted_value, confidence_score
            FROM predictions
            WHERE prediction_type='demand' AND product_name=?
            ORDER BY label ASC LIMIT 8
        """, [product_name])
        cols = ["week","forecast_qty","ci"]
        df = pd.DataFrame(r["rows"], columns=cols)
        df["forecast_qty"] = df["forecast_qty"].astype(float).round(0)
        df["ci"]           = df["ci"].astype(float).round(0)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def load_alert_summary():
    try:
        r = execute_query("""
            SELECT category, severity, COUNT(*) as n
            FROM alerts WHERE is_resolved=0
            GROUP BY category, severity
        """)
        return pd.DataFrame(r["rows"], columns=["category","severity","count"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def load_revenue_by_market():
    try:
        r = execute_query("""
            SELECT market, SUM(sales) as revenue, COUNT(*) as orders
            FROM orders GROUP BY market ORDER BY revenue DESC
        """)
        return pd.DataFrame(r["rows"], columns=["market","revenue","orders"])
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_card(label, value, sub="", color="blue"):
    color_map = {"blue":"","red":"red","amber":"amber","green":"green","grey":"grey"}
    cls = color_map.get(color, "")
    st.markdown(f"""
    <div class="kpi-card {cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def severity_badge(sev):
    cls = f"badge-{sev.lower()}" if sev else "badge-low"
    return f'<span class="{cls}">{sev}</span>'


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def alert_card(row):
    sev   = str(row.get("severity","")).lower()
    agent = row.get("agent","")
    cat   = str(row.get("category","")).upper()
    msg   = str(row.get("message",""))
    rec   = str(row.get("recommendation",""))
    ts    = str(row.get("created_at",""))[:16]
    pid   = row.get("product_name","") or ""
    oid   = row.get("order_id","")
    ref   = f"Order #{oid}" if oid else (f"Product: {pid[:40]}" if pid else "")

    short_msg = msg[:200] + ("..." if len(msg) > 200 else "")
    short_rec = rec[:150] + ("..." if len(rec) > 150 else "")

    st.markdown(f"""
    <div class="alert-card {sev}">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span>{severity_badge(sev.upper())}&nbsp;&nbsp;<strong style="font-size:13px">{cat}</strong>&nbsp;&nbsp;<span style="color:#6b7280;font-size:12px">{agent}</span></span>
        <span style="font-size:11px;color:#9ca3af">{ts}</span>
      </div>
      <div class="alert-msg">{short_msg}</div>
      <div class="alert-rec">Action: {short_rec}</div>
      <div class="alert-meta">{ref}</div>
    </div>""", unsafe_allow_html=True)


# BUG 1 FIX — removed showlegend from CHART_LAYOUT entirely.
# Passing showlegend here AND in update_layout() calls caused:
#   TypeError: update_layout() got multiple values for keyword argument 'showlegend'
# showlegend is now passed explicitly in each update_layout() call below.
PLOTLY_CONFIG = dict(displayModeBar=False)
CHART_LAYOUT  = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font_family="Inter", font_size=12,
    margin=dict(l=10, r=10, t=30, b=10),
)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.sidebar.markdown(
    "<h2 style='color:#4CAF50; font-size:42px;'>🤖 AutoBiz AI</h2>",
    unsafe_allow_html=True
)
    # st.markdown("## 🤖 AutoBiz AI")
    st.markdown("---")
    st.divider()
    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Fraud Monitor", "🚚 Delivery Risk",
         "📦 Inventory & Demand", "🔔 All Alerts"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("""
    <div style="font-size:16px;color:#6b7280;line-height:1.8">
    <b>Pipeline Status</b><br>
    M1 Preprocessing ✅<br>
    M2 ML Models ✅<br>
    M3 Database ✅<br>
    M4 Agents ✅<br>
    M5 Dashboard ✅
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.markdown("## Business Operations Dashboard")
    st.markdown("Real-time monitoring powered by AI agents · Data updated on each pipeline run")
    st.markdown("---")

    kpis = load_kpis()

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi_card("Total Orders", f"{kpis.get('total_orders',0):,}", "In database", "blue")
    with c2:
        kpi_card("Fraud Orders", f"{kpis.get('fraud_count',0):,}",
                 f"{kpis.get('fraud_count',0)/max(kpis.get('total_orders',1),1)*100:.1f}% rate", "red")
    with c3:
        kpi_card("Late Delivery Risk", f"{kpis.get('delivery_risk',0):,}",
                 "High-confidence predictions", "amber")
    with c4:
        kpi_card("Critical Stock", f"{kpis.get('critical_stock',0):,}",
                 "Products <7 days supply", "red")
    with c5:
        kpi_card("Open Alerts", f"{kpis.get('open_alerts',0):,}",
                 "Across all agents", "amber")
    with c6:
        kpi_card("Next-Week Demand", f"{kpis.get('next_week_demand',0):,}",
                 "Forecast units (all products)", "green")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Alert Summary + Revenue ───────────────────────────────────────
    col_l, col_r = st.columns([1, 1])

    with col_l:
        section("Alert Summary by Category")
        summary = load_alert_summary()
        if not summary.empty:
            sev_order = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}
            summary["sev_rank"] = summary["severity"].map(lambda x: sev_order.get(x,4))
            summary = summary.sort_values("sev_rank")
            color_map = {"CRITICAL":"#dc2626","HIGH":"#ea580c","MEDIUM":"#f59e0b","LOW":"#22c55e"}
            fig = px.bar(
                summary, x="category", y="count", color="severity",
                color_discrete_map=color_map,
                barmode="stack", text="count",
                labels={"count":"Alerts","category":"Category","severity":"Severity"},
            )
            fig.update_traces(textposition="inside", textfont_size=11)
            fig.update_layout(**CHART_LAYOUT, height=280, showlegend=True,
                              legend_orientation="h", legend_y=-0.25)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("No alert data yet. Run agents first.")

    with col_r:
        section("Revenue by Market")
        rev = load_revenue_by_market()
        if not rev.empty:
            rev["revenue_m"] = (rev["revenue"] / 1_000_000).round(2)
            fig2 = px.bar(
                rev, x="market", y="revenue_m",
                labels={"market":"Market","revenue_m":"Revenue ($M)"},
                color="revenue_m", color_continuous_scale="Blues",
                text="revenue_m",
            )
            fig2.update_traces(texttemplate="$%{text}M", textposition="outside")
            fig2.update_layout(**CHART_LAYOUT, height=280, showlegend=False,
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("No order data yet.")

    # ── Top Priority Alerts ───────────────────────────────────────────────────
    section("Top Priority Alerts")
    top_alerts = load_alerts_df(limit=6)
    if not top_alerts.empty:
        for _, row in top_alerts.head(6).iterrows():
            alert_card(row.to_dict())
    else:
        st.info("No alerts found. Run agents/run_all_agents.py first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FRAUD MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Fraud Monitor":
    st.markdown("## Fraud Monitor")
    st.markdown("Suspicious orders detected via Order Status = SUSPECTED_FRAUD · Tiered by order value")
    st.markdown("---")

    fraud_df = load_fraud_df()
    alerts_df = load_alerts_df(category="fraud", limit=500)

    # ── Stats row ─────────────────────────────────────────────────────────────
    if not fraud_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        total_fraud    = len(fraud_df)
        revenue_at_risk= fraud_df["sales"].astype(float).sum()
        critical_n     = len(alerts_df[alerts_df["severity"]=="CRITICAL"]) if not alerts_df.empty else 0
        high_n         = len(alerts_df[alerts_df["severity"]=="HIGH"])     if not alerts_df.empty else 0

        with c1: kpi_card("Fraud Orders", f"{total_fraud:,}", "In database", "red")
        with c2: kpi_card("Revenue at Risk", f"${revenue_at_risk:,.0f}", "Sum of fraud order values", "red")
        with c3: kpi_card("CRITICAL Alerts", f"{critical_n}", ">$1,000 per order", "red")
        with c4: kpi_card("HIGH Alerts",     f"{high_n}",     ">$500 per order",   "amber")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            section("Fraud Orders by Market")
            mkt = fraud_df.groupby("market").agg(
                orders=("order_id","count"),
                revenue=("sales","sum")
            ).reset_index().sort_values("revenue", ascending=False)
            fig = px.bar(mkt, x="market", y="revenue", text="orders",
                         labels={"market":"Market","revenue":"Revenue at Risk ($)","orders":"Orders"},
                         color="revenue", color_continuous_scale="Reds")
            fig.update_traces(texttemplate="%{text} orders", textposition="outside")
            fig.update_layout(**CHART_LAYOUT, height=300, showlegend=False,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        with col_r:
            section("Fraud by Customer Segment")
            seg = fraud_df.groupby("customer_segment").agg(
                count=("order_id","count"),
                revenue=("sales","sum")
            ).reset_index()
            fig2 = px.pie(seg, values="revenue", names="customer_segment",
                          color_discrete_sequence=["#ef233c","#f77f00","#fcbf49"],
                          hole=0.45)
            fig2.update_traces(textinfo="percent+label", textfont_size=12)
            fig2.update_layout(**CHART_LAYOUT, height=300, showlegend=True)
            st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Fraud Orders Table ────────────────────────────────────────────────
        section("Fraud Orders — Full List")

        # BUG 2 FIX — multiple fraud alerts can share the same order_id (e.g. one
        # CRITICAL and one HIGH alert for the same order).  Calling
        # .set_index("order_id") on a DataFrame with duplicate index values raises:
        #   ValueError: Index contains duplicate entries, cannot reshape
        # Fix: keep only the highest-priority alert per order_id before indexing.
        if not alerts_df.empty:
            sev_map = (
                alerts_df
                .drop_duplicates(subset=["order_id"], keep="first")  # already sorted by severity
                .set_index("order_id")[["severity","recommendation"]]
                .to_dict("index")
            )
        else:
            sev_map = {}

        display = fraud_df.copy()
        display["order_id"] = display["order_id"].astype(str)
        display["Severity"] = display["order_id"].map(
            lambda x: sev_map.get(int(x) if x.isdigit() else x, {}).get("severity","MEDIUM")
        )
        display["sales"] = display["sales"].astype(float).round(2)
        display = display.rename(columns={
            "order_id":"Order ID","product_name":"Product","customer_segment":"Segment",
            "market":"Market","order_region":"Region","sales":"Sale ($)",
            "delivery_status":"Delivery Status","delay_days":"Delay (days)"
        })
        show_cols = ["Order ID","Product","Segment","Market","Sale ($)","Delivery Status","Severity"]

        # Severity filter
        sev_filter = st.selectbox("Filter by Severity", ["All","CRITICAL","HIGH","MEDIUM"],
                                  key="fraud_sev")
        if sev_filter != "All":
            display = display[display["Severity"] == sev_filter]

        st.dataframe(
            display[show_cols].reset_index(drop=True),
            use_container_width=True, height=400
        )

        # ── XAI Alerts ────────────────────────────────────────────────────────
        if not alerts_df.empty:
            section("AI Explanations & Recommendations")
            st.caption("Top 10 highest-priority fraud alerts with AI-generated explanations")
            top = alerts_df.head(10)
            for _, row in top.iterrows():
                with st.expander(
                    f"🔴 Order #{row['order_id']} — {row['severity']} | "
                    f"{str(row.get('product_name',''))[:40]}",
                    expanded=False
                ):
                    st.markdown(f"**Explanation:** {row['message']}")
                    st.markdown(f"**Recommended Action:** {row['recommendation']}")
    else:
        st.info("No fraud data found. Ensure init_db.py and run_all_agents.py have been run.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DELIVERY RISK
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🚚 Delivery Risk":
    st.markdown("## Delivery Risk Monitor")
    st.markdown("RandomForest model (69.7% accuracy) · Top drivers: Shipping Mode (51%) + Lead Time (41%)")
    st.markdown("---")

    del_df    = load_delivery_df()
    alerts_df = load_alerts_df(category="delivery", limit=200)

    if not del_df.empty:
        c1, c2, c3 = st.columns(3)
        crit_n = len(del_df[del_df["risk_prob"] >= 0.85])
        high_n = len(del_df[(del_df["risk_prob"] >= 0.70) & (del_df["risk_prob"] < 0.85)])
        med_n  = len(del_df[del_df["risk_prob"] < 0.70])

        with c1: kpi_card("Critical Risk Orders", f"{crit_n}", "Prob ≥ 85%", "red")
        with c2: kpi_card("High Risk Orders",     f"{high_n}", "Prob 70–85%", "amber")
        with c3: kpi_card("Medium Risk Orders",   f"{med_n}",  "Prob 60–70%", "grey")

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([1.4, 1])

        with col_l:
            section("Risk Score Distribution")
            fig = px.histogram(
                del_df, x="risk_prob", nbins=20,
                labels={"risk_prob":"Late Delivery Probability","count":"Orders"},
                color_discrete_sequence=["#f77f00"],
            )
            fig.add_vline(x=0.85, line_dash="dash", line_color="#dc2626",
                          annotation_text="Critical (85%)", annotation_position="top right")
            fig.add_vline(x=0.70, line_dash="dash", line_color="#ea580c",
                          annotation_text="High (70%)", annotation_position="top right")
            fig.update_layout(**CHART_LAYOUT, height=280, showlegend=True)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        with col_r:
            section("Risk Tier Breakdown")
            tier_data = pd.DataFrame({
                "Tier": ["Critical (≥85%)", "High (70–85%)", "Medium (60–70%)"],
                "Count": [crit_n, high_n, med_n]
            })
            fig2 = px.pie(tier_data, values="Count", names="Tier",
                          color="Tier",
                          color_discrete_map={
                              "Critical (≥85%)":"#dc2626",
                              "High (70–85%)":"#ea580c",
                              "Medium (60–70%)":"#f59e0b"
                          }, hole=0.5)
            fig2.update_traces(textinfo="percent+label", textfont_size=11)
            fig2.update_layout(**CHART_LAYOUT, height=280, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

        # ── Orders Table ───────────────────────────────────────────────────────
        section("High-Risk Orders")
        tier_filter = st.selectbox(
            "Filter by Risk Tier",
            ["All", "Critical (≥85%)", "High (70–85%)", "Medium (60–70%)"],
            key="del_filter"
        )
        filtered = del_df.copy()
        if tier_filter == "Critical (≥85%)":
            filtered = filtered[filtered["risk_prob"] >= 0.85]
        elif tier_filter == "High (70–85%)":
            filtered = filtered[(filtered["risk_prob"] >= 0.70) & (filtered["risk_prob"] < 0.85)]
        elif tier_filter == "Medium (60–70%)":
            filtered = filtered[filtered["risk_prob"] < 0.70]

        display = filtered[["order_id","product_name","risk_pct","severity"]].rename(columns={
            "order_id":"Order ID","product_name":"Product",
            "risk_pct":"Late Risk %","severity":"Severity"
        })
        st.dataframe(display.reset_index(drop=True), use_container_width=True, height=300)

        # ── XAI Expanders ─────────────────────────────────────────────────────
        if not alerts_df.empty:
            section("AI Explanations — Top Delivery Risk Alerts")
            for _, row in alerts_df.head(8).iterrows():
                prob_val = ""
                if row.get("order_id"):
                    match = del_df[del_df["order_id"]==row["order_id"]]
                    if not match.empty:
                        prob_val = f" · {match.iloc[0]['risk_pct']}"
                with st.expander(
                    f"{'🔴' if row['severity']=='CRITICAL' else '🟠'} "
                    f"Order #{row['order_id']}{prob_val} — {str(row.get('product_name',''))[:40]}",
                    expanded=False
                ):
                    st.markdown(f"**AI Analysis:** {row['message']}")
                    st.markdown(f"---")
                    st.markdown(f"**Recommended Action:** {row['recommendation']}")
    else:
        st.info("No delivery risk predictions found. Run machinelearning.py then run_all_agents.py.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INVENTORY & DEMAND
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📦 Inventory & Demand":
    st.markdown("## Inventory & Demand")
    st.markdown("XGBoost inventory risk model · Prophet demand forecasts per product")
    st.markdown("---")

    inv_df    = load_inventory_df()
    inv_alerts = load_alerts_df(category="inventory", limit=50)
    dem_alerts = load_alerts_df(category="demand",    limit=50)

    # ── Inventory Section ─────────────────────────────────────────────────────
    section("Inventory Status")

    if not inv_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        crit = len(inv_df[inv_df["status"]=="CRITICAL"])
        warn = len(inv_df[inv_df["status"]=="WARNING"])
        ok   = len(inv_df[inv_df["status"]=="OK"])
        with c1: kpi_card("Total Products", f"{len(inv_df)}", "Tracked", "blue")
        with c2: kpi_card("Critical Stock", f"{crit}", "<7 days supply", "red")
        with c3: kpi_card("Warning Stock",  f"{warn}", "<14 days supply","amber")
        with c4: kpi_card("OK",             f"{ok}",  "Safe level",      "green")

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns([1.3, 1])

        with col_l:
            section("Days of Stock Remaining — All Products")
            chart_df = inv_df.sort_values("days_remaining").head(30).copy()
            color_map_status = {"CRITICAL":"#dc2626","WARNING":"#f59e0b","OK":"#22c55e"}
            chart_df["color"] = chart_df["status"].map(color_map_status)
            fig = px.bar(
                chart_df, x="days_remaining", y="product_name",
                orientation="h", color="status",
                color_discrete_map=color_map_status,
                labels={"days_remaining":"Days Remaining","product_name":"Product","status":"Status"},
            )
            fig.update_layout(**CHART_LAYOUT, height=420, showlegend=True,
                              yaxis_tickfont_size=10,
                              legend_orientation="h", legend_y=1.05)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        with col_r:
            section("Stock Status Distribution")
            status_counts = inv_df["status"].value_counts().reset_index()
            status_counts.columns = ["Status","Count"]
            fig2 = px.pie(
                status_counts, values="Count", names="Status",
                color="Status",
                color_discrete_map={"CRITICAL":"#dc2626","WARNING":"#f59e0b","OK":"#22c55e"},
                hole=0.5
            )
            fig2.update_traces(textinfo="percent+label", textfont_size=12)
            fig2.update_layout(**CHART_LAYOUT, height=280, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

            # Inventory alerts
            if not inv_alerts.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                section("Inventory Alerts")
                for _, row in inv_alerts.head(5).iterrows():
                    sev = str(row.get("severity","")).lower()
                    icon = "🔴" if sev=="critical" else "🟠"
                    with st.expander(f"{icon} {row.get('product_name','?')} — {row.get('severity','')}",
                                     expanded=False):
                        st.markdown(f"{row['message']}")
                        st.markdown(f"**Action:** {row['recommendation']}")

        # ── Inventory Table ───────────────────────────────────────────────────
        section("Full Inventory Table")
        status_filter = st.selectbox("Filter", ["All","CRITICAL","WARNING","OK"], key="inv_status")
        display_inv = inv_df.copy() if status_filter=="All" else inv_df[inv_df["status"]==status_filter]
        display_inv = display_inv[[
            "product_name","category","simulated_stock","avg_daily_demand",
            "days_remaining","reorder_threshold","product_price","status"
        ]].rename(columns={
            "product_name":"Product","category":"Category",
            "simulated_stock":"Stock (units)","avg_daily_demand":"Demand/Day",
            "days_remaining":"Days Left","reorder_threshold":"Reorder Threshold",
            "product_price":"Unit Price ($)","status":"Status"
        })
        display_inv["Days Left"] = display_inv["Days Left"].astype(float).round(1)
        display_inv["Demand/Day"] = display_inv["Demand/Day"].astype(float).round(2)
        st.dataframe(display_inv.reset_index(drop=True), use_container_width=True, height=350)
    else:
        st.info("No inventory data found. Run init_db.py first.")

    st.markdown("---")

    # ── Demand Forecast Section ───────────────────────────────────────────────
    section("Demand Forecast — Product Lookup")

    try:
        prod_r = execute_query(
            "SELECT DISTINCT product_name FROM predictions "
            "WHERE prediction_type='demand' ORDER BY product_name LIMIT 118"
        )
        products = [r[0] for r in prod_r["rows"] if r[0]]
    except Exception:
        products = []

    if products:
        col_sel, _ = st.columns([2, 3])
        with col_sel:
            selected_product = st.selectbox("Select Product", products, key="prod_select")

        if selected_product:
            fc_df = load_demand_forecast(selected_product)
            if not fc_df.empty:
                col_chart, col_detail = st.columns([2, 1])
                with col_chart:
                    section(f"8-Week Forecast · {selected_product[:45]}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fc_df["week"], y=fc_df["forecast_qty"],
                        mode="lines+markers", name="Forecast",
                        line=dict(color="#4361ee", width=2),
                        marker=dict(size=7),
                    ))
                    fig.add_trace(go.Scatter(
                        x=fc_df["week"],
                        y=fc_df["forecast_qty"] + fc_df["ci"]/2,
                        mode="lines", line=dict(width=0), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=fc_df["week"],
                        y=(fc_df["forecast_qty"] - fc_df["ci"]/2).clip(lower=0),
                        mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor="rgba(67,97,238,0.1)",
                        name="Confidence Range", showlegend=True
                    ))
                    fig.update_layout(**CHART_LAYOUT, height=300, showlegend=True,
                                      xaxis_title="Week", yaxis_title="Units")
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

                with col_detail:
                    section("Weekly Breakdown")
                    fc_display = fc_df[["week","forecast_qty"]].rename(
                        columns={"week":"Week","forecast_qty":"Forecast (units)"}
                    )
                    st.dataframe(fc_display, use_container_width=True, height=280)
            else:
                st.info(f"No forecast data for '{selected_product}'.")
    else:
        st.info("No demand forecasts in DB. Run machinelearning.py first.")

    # ── Demand Alerts ─────────────────────────────────────────────────────────
    if not dem_alerts.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        section("Demand Alerts")
        col_s, col_d = st.columns(2)
        surge_alerts   = dem_alerts[dem_alerts["severity"]=="HIGH"]
        decline_alerts = dem_alerts[dem_alerts["severity"]=="MEDIUM"]

        with col_s:
            st.markdown("**📈 Demand Surges**")
            if not surge_alerts.empty:
                for _, row in surge_alerts.head(5).iterrows():
                    with st.expander(f"🟠 {str(row.get('product_name','?'))[:40]}", expanded=False):
                        st.markdown(row["message"])
                        st.markdown(f"**Action:** {row['recommendation']}")
            else:
                st.caption("No surge alerts.")

        with col_d:
            st.markdown("**📉 Demand Declines**")
            if not decline_alerts.empty:
                for _, row in decline_alerts.head(5).iterrows():
                    with st.expander(f"🟡 {str(row.get('product_name','?'))[:40]}", expanded=False):
                        st.markdown(row["message"])
                        st.markdown(f"**Action:** {row['recommendation']}")
            else:
                st.caption("No decline alerts.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ALL ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔔 All Alerts":
    st.markdown("## Alert Centre")
    st.markdown("All active alerts from every agent · Filter, review, and resolve")
    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cat_filter = st.selectbox(
            "Category", ["All","fraud","delivery","inventory","demand","conflict"], key="ac_cat"
        )
    with col2:
        sev_filter = st.selectbox(
            "Severity", ["All","CRITICAL","HIGH","MEDIUM"], key="ac_sev"
        )
    
    with col3:
        limit_n = st.selectbox("Show", [50, 100, 200, 500], key="ac_lim")
    with col4:
        show_resolved = st.checkbox("Include Resolved", value=False, key="ac_res")
        
    cat_arg = None if cat_filter=="All" else cat_filter
    sev_arg = None if sev_filter=="All" else sev_filter
    all_df  = load_alerts_df(
        category=cat_arg, severity=sev_arg,
        resolved=show_resolved, limit=limit_n
    )

    # ── Summary badges ────────────────────────────────────────────────────────
    if not all_df.empty:
        total     = len(all_df)
        n_crit    = len(all_df[all_df["severity"]=="CRITICAL"])
        n_high    = len(all_df[all_df["severity"]=="HIGH"])
        n_med     = len(all_df[all_df["severity"]=="MEDIUM"])
        n_resolved= len(all_df[all_df["is_resolved"]==1])

        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: kpi_card("Total Shown",  f"{total}",      "", "blue")
        with c2: kpi_card("Critical",     f"{n_crit}",     "", "red")
        with c3: kpi_card("High",         f"{n_high}",     "", "amber")
        with c4: kpi_card("Medium",       f"{n_med}",      "", "grey")
        with c5: kpi_card("Resolved",     f"{n_resolved}", "", "green")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Alert Cards ───────────────────────────────────────────────────────
        section(f"Alerts ({total} shown)")

        for _, row in all_df.iterrows():
            row_dict = row.to_dict()
            col_card, col_action = st.columns([5, 1])

            with col_card:
                alert_card(row_dict)

            with col_action:
                alert_id = row_dict.get("id")
                is_res   = row_dict.get("is_resolved", 0)
                if not is_res and alert_id:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("✅ Resolve", key=f"res_{alert_id}",
                                 use_container_width=True):
                        try:
                            resolve_alert(int(alert_id))
                            st.cache_data.clear()
                            st.success("Resolved")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                elif is_res:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        '<div style="font-size:11px;color:#22c55e;padding:8px 0">✓ Resolved</div>',
                        unsafe_allow_html=True
                    )
    else:
        st.info("No alerts match the current filters.")

    # ── XAI Detail Expanders ──────────────────────────────────────────────────
    if not all_df.empty:
        section("Detailed AI Explanations")
        st.caption("Expand any alert to read the full AI explanation and recommended action")
        for _, row in all_df.head(20).iterrows():
            sev  = str(row.get("severity",""))
            cat  = str(row.get("category","")).upper()
            pid  = str(row.get("product_name","") or "")
            oid  = row.get("order_id","")
            ref  = f"Order #{oid}" if oid else (pid[:35] if pid else "General")
            icon = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡"}.get(sev, "⚪")
            with st.expander(f"{icon} [{cat}] {ref} — {sev}", expanded=False):
                st.markdown(f"**Agent:** {row.get('agent','')}  |  "
                            f"**Time:** {str(row.get('created_at',''))[:16]}")
                st.markdown("---")
                st.markdown(f"**AI Analysis:**  \n{row.get('message','')}")
                st.markdown("---")
                st.markdown(f"**Recommended Action:**  \n{row.get('recommendation','')}")
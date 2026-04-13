# =============================================================================
# AGENT: FRAUD DETECTION AGENT  —  agents/fraud_agent.py
#
# Reads fraud orders from orders table (fraud_label=1).
# Collects ALL alerts in memory, then inserts in one batched call.
# This avoids WinError 10053 (Windows killing rapid successive HTTP connections).
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import execute_query, insert_alerts_batch

HIGH_VALUE_THRESHOLD     = 500.0
CRITICAL_VALUE_THRESHOLD = 1000.0


def _get_fraud_orders(limit=500):
    try:
        result = execute_query("""
            SELECT order_id, order_date, product_name, customer_segment,
                   market, order_region, sales, delivery_status,
                   delay_days, shipping_mode, department
            FROM orders WHERE fraud_label = 1
            ORDER BY sales DESC LIMIT ?
        """, [limit])
        cols = ["order_id","order_date","product_name","customer_segment",
                "market","order_region","sales","delivery_status",
                "delay_days","shipping_mode","department"]
        return [dict(zip(cols, row)) for row in result["rows"]]
    except Exception as e:
        print(f"  ERROR reading fraud orders: {e}")
        return []


def _build_alert(order):
    oid      = order.get("order_id", "?")
    product  = order.get("product_name", "Unknown product")
    sales    = float(order.get("sales") or 0)
    market   = order.get("market", "Unknown")
    segment  = order.get("customer_segment", "Unknown")
    region   = order.get("order_region", "")
    ship     = order.get("shipping_mode", "Unknown")
    delay    = float(order.get("delay_days") or 0)
    delivery = str(order.get("delivery_status") or "")
    dept     = order.get("department", "")

    ctx = [f"${sales:,.2f}", f"{segment}", f"{market} ({region})", dept]
    if "canceled" in delivery.lower():
        ctx.append("shipping cancelled after placement")
    elif delay > 1:
        ctx.append(f"delayed {delay:.0f} days")

    if sales >= CRITICAL_VALUE_THRESHOLD:
        severity = "CRITICAL"
        rec = (f"IMMEDIATE: Hold Order #{oid}. Escalate to finance for identity "
               f"and payment verification. Do NOT ship until cleared. "
               f"If dispatched, initiate courier intercept.")
    elif sales >= HIGH_VALUE_THRESHOLD:
        severity = "HIGH"
        rec = (f"HOLD Order #{oid}. Contact {segment} customer via registered "
               f"details to verify. Require payment confirmation within 4 hours.")
    elif "canceled" in delivery.lower():
        severity = "MEDIUM"
        rec = (f"INVESTIGATE Order #{oid}: Check if fraudulent refund/chargeback "
               f"initiated. Flag customer account for monitoring.")
    else:
        severity = "MEDIUM"
        rec = (f"MONITOR Order #{oid}: Flag for watchlist. Verify customer history "
               f"before processing future orders. Escalate if pattern repeats.")

    msg = (f"Order #{oid} for '{product}' flagged as SUSPECTED FRAUD. "
           f"Profile: {', '.join(str(c) for c in ctx if c)}. Shipping: {ship}. "
           f"Requires manual review before any fulfilment action.")

    return {
        "agent"         : "FraudAgent",
        "severity"      : severity,
        "category"      : "fraud",
        "message"       : msg,
        "recommendation": rec,
        "order_id"      : oid,
        "product_name"  : product,
    }


def run_fraud_agent():
    print("\n" + "=" * 55)
    print("  FraudAgent — Running")
    print("=" * 55)

    orders = _get_fraud_orders(limit=500)
    if not orders:
        print("  No fraud orders found. Run init_db.py first.")
        return

    print(f"  Found {len(orders)} fraud orders — building alerts...")

    alerts     = [_build_alert(o) for o in orders]
    critical_n = sum(1 for a in alerts if a["severity"] == "CRITICAL")
    high_n     = sum(1 for a in alerts if a["severity"] == "HIGH")
    medium_n   = sum(1 for a in alerts if a["severity"] == "MEDIUM")

    print(f"  Saving {len(alerts)} alerts to DB (batched)...")
    insert_alerts_batch(alerts, chunk_size=50)

    print(f"\n  Alerts saved:")
    print(f"    CRITICAL (>${CRITICAL_VALUE_THRESHOLD:,.0f}): {critical_n}")
    print(f"    HIGH     (>${HIGH_VALUE_THRESHOLD:,.0f}):   {high_n}")
    print(f"    MEDIUM   (rest):            {medium_n}")
    print(f"  FraudAgent complete.")


if __name__ == "__main__":
    run_fraud_agent()

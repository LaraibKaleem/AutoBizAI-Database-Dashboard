# =============================================================================
# AGENT: LATE DELIVERY RISK AGENT  —  agents/delivery_agent.py
#
# Reads top-200 high-risk delivery predictions from DB.
# Enriches from clean_data.csv (not DB orders table — only has 10K rows).
# Batches all alert inserts in one call to avoid connection errors.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from modules.database import execute_query, insert_alerts_batch

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA = os.path.join(BASE_DIR, "data", "processed", "clean_data.csv")

SHIPPING_RISK = {
    "First Class"    : ("95%", "First Class has the highest late rate (95.3%) despite the premium label."),
    "Second Class"   : ("77%", "Second Class has a 76.6% late delivery rate."),
    "Same Day"       : ("46%", "Nearly half of Same Day orders miss the commitment (45.7% late rate)."),
    "Standard Class" : ("38%", "Standard Class has the lowest late rate (38.1%)."),
}


def _load_order_lookup():
    try:
        df = pd.read_csv(CLEAN_DATA, usecols=[
            "Order Id", "Product Name", "Shipping Mode", "Market",
            "Order Region", "Customer Segment", "Sales",
            "Department Name", "Days for shipment (scheduled)"
        ])
        return {
            int(row["Order Id"]): {
                "product_name"  : row.get("Product Name", ""),
                "shipping_mode" : row.get("Shipping Mode", ""),
                "market"        : row.get("Market", ""),
                "order_region"  : row.get("Order Region", ""),
                "segment"       : row.get("Customer Segment", ""),
                "sales"         : float(row.get("Sales") or 0),
                "department"    : row.get("Department Name", ""),
                "scheduled_days": float(row.get("Days for shipment (scheduled)") or 0),
            }
            for _, row in df.iterrows()
        }
    except Exception as e:
        print(f"  Warning: could not load enrichment data: {e}")
        return {}


def _build_alert(pred, ctx, order_id):
    prob = float(pred.get("predicted_value") or 0)

    if prob >= 0.85:
        severity = "CRITICAL"
    elif prob >= 0.70:
        severity = "HIGH"
    else:
        severity = "MEDIUM"

    if not ctx:
        msg = (f"LATE DELIVERY RISK — Order #{order_id}: "
               f"Model probability of late arrival: {prob*100:.0f}%.")
        rec = (f"Monitor Order #{order_id}. Escalate if no tracking "
               f"movement within 24 hours of dispatch.")
    else:
        product = ctx.get("product_name", "Unknown")
        ship    = ctx.get("shipping_mode", "Unknown")
        market  = ctx.get("market", "")
        region  = ctx.get("order_region", "")
        segment = ctx.get("segment", "")
        sales   = ctx.get("sales", 0)
        dept    = ctx.get("department", "")
        sched   = ctx.get("scheduled_days", 0)

        rate, reason = SHIPPING_RISK.get(ship, ("?%", "Unknown shipping mode."))
        lead = (f"{sched:.0f}-day lead time — very tight." if sched <= 2
                else f"{sched:.0f}-day lead time.")

        msg = (f"LATE DELIVERY RISK — Order #{order_id} for '{product}' "
               f"({dept}, {market}/{region}): {prob*100:.0f}% late probability. "
               f"Shipping Mode (51% of signal): '{ship}' has {rate} late rate. {reason} "
               f"Lead Time (41% of signal): {lead} "
               f"Order value: ${sales:,.2f} | Customer: {segment}.")

        if prob >= 0.85:
            rec = (f"URGENT Order #{order_id}: Proactively contact {segment} customer. "
                   f"Consider upgrade from '{ship}'. Review carrier performance on this route.")
        elif prob >= 0.70:
            rec = (f"ACTION Order #{order_id}: Monitor closely once dispatched. "
                   f"Prepare delay notification template. Evaluate '{ship}' carrier.")
        else:
            rec = (f"WATCH Order #{order_id}: Include in daily shipment monitoring. "
                   f"Escalate if no movement within 24 hours of dispatch.")

    return {
        "agent"         : "DeliveryAgent",
        "severity"      : severity,
        "category"      : "delivery",
        "message"       : msg,
        "recommendation": rec,
        "order_id"      : order_id,
        "product_name"  : (ctx or {}).get("product_name"),
    }


def run_delivery_agent():
    print("\n" + "=" * 55)
    print("  DeliveryAgent — Running")
    print("=" * 55)

    try:
        result = execute_query("""
            SELECT id, product_name, order_id, predicted_value, confidence_score, label
            FROM predictions
            WHERE prediction_type = 'delivery' AND label = 'late'
            ORDER BY predicted_value DESC LIMIT 200
        """)
        cols  = ["id","product_name","order_id","predicted_value","confidence_score","label"]
        preds = [dict(zip(cols, row)) for row in result["rows"]]
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    if not preds:
        print("  No delivery predictions in DB. Run machinelearning.py first.")
        return

    print(f"  Found {len(preds)} high-risk predictions")
    print(f"  Loading order enrichment from clean_data.csv...")
    lookup = _load_order_lookup()
    print(f"  Enrichment loaded: {len(lookup):,} orders")

    alerts = []
    for pred in preds:
        oid = pred.get("order_id")
        ctx = lookup.get(int(oid)) if oid else None
        alerts.append(_build_alert(pred, ctx, oid))

    counts = {}
    for a in alerts:
        counts[a["severity"]] = counts.get(a["severity"], 0) + 1

    print(f"  Saving {len(alerts)} alerts (batched)...")
    insert_alerts_batch(alerts, chunk_size=50)

    print(f"\n  Alerts saved: {counts}")
    print(f"  DeliveryAgent complete.")


if __name__ == "__main__":
    run_delivery_agent()

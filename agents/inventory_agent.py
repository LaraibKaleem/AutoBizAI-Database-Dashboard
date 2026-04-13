# =============================================================================
# AGENT: INVENTORY AGENT  —  agents/inventory_agent.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import get_predictions, get_inventory_status, insert_alerts_batch

LEAD_TIME_DAYS       = 5
CRITICAL_DAYS_BUFFER = 14
LOW_DAYS_BUFFER      = 30


def _reorder_qty(inv, label):
    try:
        daily  = float(inv.get("avg_daily_demand") or 0)
        stock  = float(inv.get("simulated_stock") or 0)
        buf    = CRITICAL_DAYS_BUFFER if label == "critical" else LOW_DAYS_BUFFER
        return max(int(round((daily * (buf + LEAD_TIME_DAYS)) - stock)), 1)
    except Exception:
        return 0


def _build_alert(label, pname, inv):
    if not inv:
        if label == "critical":
            msg = (f"CRITICAL STOCK ALERT — '{pname}': "
                   f"Immediate reorder required. Stock critically low.")
            rec = f"URGENT: Contact supplier NOW for expedited delivery of '{pname}'."
        else:
            msg = (f"LOW STOCK WARNING — '{pname}': "
                   f"Stock approaching minimum threshold.")
            rec = f"PLAN REORDER: Place standard PO for '{pname}' within 48 hours."
        return {
            "agent": "InventoryAgent",
            "severity": "CRITICAL" if label == "critical" else "HIGH",
            "category": "inventory",
            "message": msg, "recommendation": rec,
            "product_name": pname,
        }

    stock    = int(float(inv.get("simulated_stock") or 0))
    daily    = float(inv.get("avg_daily_demand") or 0)
    days_rem = float(inv.get("days_remaining") or 0)
    thresh   = float(inv.get("reorder_threshold") or 0)
    price    = float(inv.get("product_price") or 0)
    category = inv.get("category", "")
    qty      = _reorder_qty(inv, label)

    if label == "critical":
        severity = "CRITICAL"
        msg = (f"CRITICAL STOCK — '{pname}' ({category}): "
               f"{stock} units left. At {daily:.1f}/day, runs out in {days_rem:.0f} days. "
               f"Reorder threshold: {thresh:.0f} units. "
               f"Order {qty} units now ({CRITICAL_DAYS_BUFFER}d cover + {LEAD_TIME_DAYS}d lead). "
               f"Est. cost: ${qty * price:,.2f} at ${price:.2f}/unit.")
        rec = (f"URGENT: Contact supplier for expedited {LEAD_TIME_DAYS}-day delivery. "
               f"Order {qty} units of '{pname}'. "
               f"Activate secondary supplier if primary cannot deliver in time.")
    else:
        severity = "HIGH"
        msg = (f"LOW STOCK — '{pname}' ({category}): "
               f"{stock} units, {days_rem:.0f} days supply at {daily:.1f}/day. "
               f"Reorder threshold: {thresh:.0f} units. "
               f"Order {qty} units ({LOW_DAYS_BUFFER}d cover). "
               f"Est. cost: ${qty * price:,.2f}.")
        rec = (f"Place standard PO for {qty} units of '{pname}'. "
               f"Delivery in {LEAD_TIME_DAYS} business days. "
               f"Monitor daily until stock replenished.")

    return {
        "agent": "InventoryAgent", "severity": severity, "category": "inventory",
        "message": msg, "recommendation": rec, "product_name": pname,
    }


def run_inventory_agent():
    print("\n" + "=" * 55)
    print("  InventoryAgent — Running")
    print("=" * 55)

    predictions = get_predictions("inventory", limit=200)
    if not predictions:
        print("  No inventory predictions. Run machinelearning.py first.")
        return

    print(f"  Found {len(predictions)} inventory predictions")

    try:
        inv_df  = get_inventory_status()
        inv_map = {str(r["product_name"]): dict(r) for _, r in inv_df.iterrows()}
    except Exception as e:
        print(f"  Warning: inventory table unavailable ({e})")
        inv_map = {}

    alerts = []
    safe_n = 0

    for pred in predictions:
        if isinstance(pred, dict):
            label = (pred.get("label") or "safe").lower()
            pname = str(pred.get("product_name") or "")
        else:
            label = str(pred[7] or "safe").lower()
            pname = str(pred[3] or "")

        if label == "safe":
            safe_n += 1
            continue

        alerts.append(_build_alert(label, pname, inv_map.get(pname)))

    critical_n = sum(1 for a in alerts if a["severity"] == "CRITICAL")
    high_n     = sum(1 for a in alerts if a["severity"] == "HIGH")

    if alerts:
        print(f"  Saving {len(alerts)} alerts (batched)...")
        insert_alerts_batch(alerts, chunk_size=50)

    print(f"\n  Alerts saved:")
    print(f"    CRITICAL: {critical_n}")
    print(f"    HIGH (low stock): {high_n}")
    print(f"    Skipped (safe): {safe_n}")
    print(f"  InventoryAgent complete.")


if __name__ == "__main__":
    run_inventory_agent()

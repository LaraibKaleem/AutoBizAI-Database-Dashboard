# =============================================================================
# AGENT: CONFLICT RESOLVER  —  agents/conflict_resolver.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import get_alerts, insert_alerts_batch, execute_query

SEVERITY_RANK = {"LOW":1, "MEDIUM":2, "HIGH":3, "CRITICAL":4}


def _to_dict(a):
    if isinstance(a, dict):
        return a
    return {"id":a[0],"created_at":a[1],"agent":a[2],"severity":a[3],
            "category":a[4],"product_name":a[5],"order_id":a[6],
            "message":a[7],"recommendation":a[8],"is_resolved":a[9]}


def run_conflict_resolver():
    print("\n" + "=" * 55)
    print("  ConflictResolver — Running")
    print("=" * 55)

    raw = get_alerts(resolved=False, limit=2000)
    if not raw:
        print("  No open alerts. Run agents first.")
        return

    alerts = [_to_dict(a) for a in raw]
    print(f"  Analysing {len(alerts)} open alerts...")

    new_alerts = []

    # ── By product_name ───────────────────────────────────────────────────
    by_product = {}
    for a in alerts:
        p = a.get("product_name")
        if p:
            by_product.setdefault(p, []).append(a)

    for product, palerts in by_product.items():
        cats = {a["category"] for a in palerts}

        # Conflict: Inventory says restock BUT Demand says decline
        if "inventory" in cats and "demand" in cats:
            inv_a = next((a for a in palerts if a["category"]=="inventory"), None)
            dem_a = next((a for a in palerts if a["category"]=="demand"), None)
            if inv_a and dem_a and "decline" in str(dem_a.get("message","")).lower():
                inv_sev = inv_a.get("severity","HIGH")
                if SEVERITY_RANK.get(inv_sev,0) >= 3:
                    msg = (f"CONFLICT — '{product}': InventoryAgent raised {inv_sev} restock "
                           f"alert BUT DemandAgent forecasts declining demand. "
                           f"Resolution: partial reorder only — cover lead time + 7 days, "
                           f"not the standard 30-day buffer.")
                    rec = (f"PARTIAL REORDER for '{product}': Order 7 days of supply only. "
                           f"Re-evaluate demand in 2 weeks before committing to full reorder.")
                    sev = "HIGH"
                else:
                    msg = (f"CONFLICT — '{product}': Low stock flagged but demand declining. "
                           f"Deferring reorder is the safer option.")
                    rec = (f"DEFER REORDER for '{product}': Monitor 2 weeks. "
                           f"Reorder only if demand stabilises or stock hits critical.")
                    sev = "MEDIUM"
                new_alerts.append({"agent":"ConflictResolver","severity":sev,
                    "category":"conflict","message":msg,"recommendation":rec,
                    "product_name":product})

        # Compound: Delivery risk + Inventory shortage on same product
        if "delivery" in cats and "inventory" in cats and "fraud" not in cats:
            msg = (f"COMPOUND RISK — '{product}': Late delivery risk AND inventory "
                   f"shortage flagged. If delivery is delayed AND stock runs out, "
                   f"fulfilment will be impossible.")
            rec = (f"ESCALATE '{product}': Expedite shipping AND trigger emergency "
                   f"restock simultaneously. Notify both warehouse and operations managers.")
            new_alerts.append({"agent":"ConflictResolver","severity":"HIGH",
                "category":"conflict","message":msg,"recommendation":rec,
                "product_name":product})

    # ── By order_id ───────────────────────────────────────────────────────
    by_order = {}
    for a in alerts:
        oid = a.get("order_id")
        if oid is not None:
            by_order.setdefault(str(oid), []).append(a)

    for order_id, oalerts in by_order.items():
        cats   = {a["category"] for a in oalerts}
        agents = {a["agent"] for a in oalerts}
        if "fraud" in cats and len(cats) > 1:
            other = cats - {"fraud"}
            other_agents = {a["agent"] for a in oalerts if a["category"] != "fraud"}
            msg = (f"ESCALATION — Order #{order_id}: Flagged by FraudAgent AND "
                   f"{', '.join(other_agents)} ({', '.join(other)}). "
                   f"Multiple independent risk signals confirm high-risk transaction.")
            rec = (f"BLOCK ORDER #{order_id}: Suspend immediately. Do not process "
                   f"until cleared by fraud team and operations manager.")
            new_alerts.append({"agent":"ConflictResolver","severity":"CRITICAL",
                "category":"conflict","message":msg,"recommendation":rec,
                "order_id":order_id})

    if new_alerts:
        print(f"  Saving {len(new_alerts)} conflict alerts (batched)...")
        insert_alerts_batch(new_alerts, chunk_size=50)

    print(f"  Conflicts resolved: {len(new_alerts)}")
    print(f"  ConflictResolver complete.")


def generate_top_recommendations(top_n=5):
    print("\n" + "=" * 55)
    print("  TOP PRIORITY ACTIONS — Manager Daily Briefing")
    print("=" * 55)

    output = []
    for sev in ["CRITICAL", "HIGH"]:
        for a in get_alerts(severity=sev, resolved=False, limit=top_n*2):
            output.append(_to_dict(a))
        if len(output) >= top_n:
            break

    if not output:
        print("  No critical or high alerts open.")
        return

    for i, a in enumerate(output[:top_n], 1):
        msg = str(a.get("message",""))
        rec = str(a.get("recommendation",""))
        print(f"\n  [{i}] {a['severity']} | {str(a['category']).upper()} | {a['agent']}")
        print(f"  Issue : {msg[:200]}{'...' if len(msg)>200 else ''}")
        print(f"  Action: {rec[:200]}{'...' if len(rec)>200 else ''}")


if __name__ == "__main__":
    run_conflict_resolver()
    generate_top_recommendations()

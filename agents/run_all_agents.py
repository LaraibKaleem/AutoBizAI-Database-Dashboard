# =============================================================================
# MODULE 4 — RUN ALL AGENTS  —  agents/run_all_agents.py
#
# Run order:
#   1. FraudAgent       → fraud_label=1 orders     → fraud alerts
#   2. DeliveryAgent    → delivery ML predictions  → late-risk alerts
#   3. InventoryAgent   → inventory ML predictions → stock alerts
#   4. DemandAgent      → Prophet forecasts        → demand alerts
#   5. ConflictResolver → all alerts               → conflict alerts
#
# Each agent collects alerts in memory, then batch-inserts in one call.
# Individual agent failures are caught — pipeline continues regardless.
# =============================================================================

import sys, os, traceback, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import execute_query


def _run(name, fn, cooldown=3):
    """Run one agent, catch errors, then pause to let TCP connections settle."""
    try:
        fn()
        ok = True
    except Exception as e:
        print(f"\n  ERROR in {name}: {e}")
        traceback.print_exc()
        ok = False
    print(f"  Pausing {cooldown}s before next agent...")
    time.sleep(cooldown)
    return ok


if __name__ == "__main__":
    from agents.fraud_agent       import run_fraud_agent
    from agents.delivery_agent    import run_delivery_agent
    from agents.inventory_agent   import run_inventory_agent
    from agents.demand_agent      import run_demand_agent
    from agents.conflict_resolver import run_conflict_resolver, generate_top_recommendations

    print("\n" + "#" * 55)
    print("  AutoBiz AI — MODULE 4: All Agents")
    print("#" * 55)

    try:
        execute_query("SELECT 1")
        print("  DB: Connected")
    except Exception as e:
        print(f"  ERROR: DB unreachable: {e}")
        exit(1)

    # Clear all previous alerts BEFORE running agents
    # (insert_alerts_batch does bulk-delete by agent+category, so this ensures
    #  a completely clean slate even if an agent was renamed or removed)
    try:
        execute_query("DELETE FROM alerts")
        print("  Previous alerts cleared.")
    except Exception as e:
        print(f"  Warning: could not clear alerts: {e}")

    results = {}
    results["FraudAgent"]       = _run("FraudAgent",       run_fraud_agent)
    results["DeliveryAgent"]    = _run("DeliveryAgent",     run_delivery_agent)
    results["InventoryAgent"]   = _run("InventoryAgent",    run_inventory_agent)
    results["DemandAgent"]      = _run("DemandAgent",       run_demand_agent)
    results["ConflictResolver"] = _run("ConflictResolver",  run_conflict_resolver)

    generate_top_recommendations()

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)

    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")

    print()
    try:
        for cat in ["fraud","delivery","inventory","demand","conflict"]:
            r = execute_query(
                "SELECT severity, COUNT(*) FROM alerts "
                "WHERE category=? AND is_resolved=0 GROUP BY severity",
                [cat]
            )
            counts = {row[0]: int(row[1]) for row in r["rows"]}
            total  = sum(counts.values())
            if total:
                print(f"  {cat.upper():12s}: {total:3d}  {counts}")
        r = execute_query("SELECT COUNT(*) FROM alerts WHERE is_resolved=0")
        print(f"  {'TOTAL':12s}: {int(r['rows'][0][0]):3d} open alerts")
    except Exception as e:
        print(f"  Summary error: {e}")

    print("\n  Next: streamlit run app.py")
    print("#" * 55)

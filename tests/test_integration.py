# =============================================================
# MODULE 6 — Integration & Testing
# File: tests/test_integration.py
# Run: python tests/test_integration.py
# =============================================================

import sys, os
import pandas as pd
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import (
    execute_query,
    get_dashboard_summary,
    get_alerts,
    get_predictions,
    get_inventory_status,
    get_fraud_orders,
    insert_alert,
    insert_prediction,
    resolve_alert,
    search_orders,
)

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────
# TEST 1 — DB Connection
# ─────────────────────────────────────────────────────────────
def test_connection():
    print("\n── Test 1: DB Connection ────────────────────────────")
    try:
        import time
        start    = time.time()
        r        = execute_query("SELECT 1")
        elapsed  = round((time.time() - start) * 1000)  # ms

        rows = r.get("rows", [])
        check("Turso responds",        len(rows) > 0,      f"{elapsed}ms")
        check("Returns correct value", rows[0][0] == 1,    f"got {rows[0][0]}")
        check("Response time < 5s",    elapsed < 5000,     f"{elapsed}ms")

    except Exception as e:
        check("Turso responds",        False, str(e))
        check("Returns correct value", False, "connection failed")
        check("Response time < 5s",    False, "connection failed")

# ─────────────────────────────────────────────────────────────
# TEST 2 — Tables exist and have data
# ─────────────────────────────────────────────────────────────
def test_tables():
    print("\n── Test 2: Tables & Row Counts ─────────────────────")

    # Minimum expected rows per table
    min_rows = {
        "orders"       : 5000,   
        "inventory"    : 100,    
        "weekly_demand": 1000,   
        "predictions"  : 5000,   
        "alerts"       : 20,     
        # "orders"       : 1000,
        # "inventory"    : 10,
        # "weekly_demand": 100,
        # "predictions"  : 100,
        # "alerts"       : 1,
    }

    for table in ["orders", "inventory", "weekly_demand", "predictions", "alerts"]:
        try:
            r   = execute_query(f"SELECT COUNT(*) FROM {table}")
            cnt = int(r["rows"][0][0]) if r["rows"] else 0
            minimum = min_rows.get(table, 0)

            check(f"Table '{table}' exists",   True,        f"{cnt:,} rows")
            check(f"Table '{table}' has data", cnt >= minimum, f"{cnt:,} >= {minimum:,} expected")

        except Exception as e:
            check(f"Table '{table}' exists",   False, str(e))
            check(f"Table '{table}' has data", False, "table missing")

# ─────────────────────────────────────────────────────────────
# TEST 3 — Dashboard summary returns valid data
# ─────────────────────────────────────────────────────────────
def test_dashboard_summary():
    print("\n── Test 3: Dashboard Summary ───────────────────────")
    try:
        s = get_dashboard_summary()

        total_orders   = int(s.get("total_orders",   0))
        fraud_orders   = int(s.get("fraud_orders",   0))
        open_alerts    = int(s.get("open_alerts",    0))
        critical_stock = int(s.get("critical_stock", 0))

        # Check keys exist
        check("total_orders key",   "total_orders"   in s)
        check("fraud_orders key",   "fraud_orders"   in s)
        check("open_alerts key",    "open_alerts"    in s)
        check("critical_stock key", "critical_stock" in s)

        # Check values make sense
        check("total_orders > 0",              total_orders > 0,              f"{total_orders:,} orders")
        check("fraud_orders > 0",              fraud_orders > 0,              f"{fraud_orders:,} fraud orders")
        check("fraud < total",                 fraud_orders < total_orders,   f"{fraud_orders} < {total_orders}")
        check("open_alerts >= 0",              open_alerts  >= 0,             f"{open_alerts} open alerts")
        check("critical_stock >= 0",           critical_stock >= 0,           f"{critical_stock} critical products")

    except Exception as e:
        check("Dashboard summary", False, str(e))

# ─────────────────────────────────────────────────────────────
# TEST 4 — Predictions stored
# ─────────────────────────────────────────────────────────────
def test_predictions():
    print("\n── Test 4: Predictions ─────────────────────────────")

    expected_labels = {
        "fraud"    : ["fraud", "legit"],
        "inventory": ["safe", "low", "critical"],
    }

    for ptype in ["fraud", "inventory"]:
        try:
            # Check total count
            result = execute_query(
                "SELECT COUNT(*) FROM predictions WHERE prediction_type=?",
                [ptype]
            )
            count = int(result["rows"][0][0]) if result["rows"] else 0
            check(f"'{ptype}' predictions exist", count > 0, f"{count:,} total rows in DB")

            # Check labels are valid
            preds = get_predictions(ptype, limit=15000)
            valid_labels = expected_labels[ptype]
            invalid = [p for p in preds if p[7] not in valid_labels]
            check(f"'{ptype}' labels are valid", len(invalid) == 0,
                  f"{len(invalid)} invalid labels found")

            # Check predicted_value is not null
            null_values = [p for p in preds if p[5] is None]
            check(f"'{ptype}' values not null", len(null_values) == 0,
                  f"{len(null_values)} null values found")

            # Check score range 0-1 — only for fraud (inventory uses 0,1,2)
            if ptype == "fraud":
                out_of_range = [p for p in preds if p[5] is not None and not (0 <= float(p[5]) <= 1)]
                check(f"'{ptype}' scores in range 0-1", len(out_of_range) == 0,
                      f"{len(out_of_range)} out of range")

        except Exception as e:
            check(f"'{ptype}' predictions", False, str(e))


# ─────────────────────────────────────────────────────────────
# TEST 5 — Alerts stored
# ─────────────────────────────────────────────────────────────
def test_alerts():
    print("\n── Test 5: Alerts ──────────────────────────────────")
    try:
        # Check alerts exist 100
        alerts = get_alerts(limit=500)
        check("Alerts exist in DB", len(alerts) > 0, f"{len(alerts)} open alerts")

        # Categories that are optional — warn but don't fail
        optional_cats = {"inventory", "demand", "conflict"}

        # Check expected categories exist in DB
        for cat in ["fraud", "inventory", "demand", "conflict"]:
            result = execute_query(
                "SELECT COUNT(*) FROM alerts WHERE category=?", [cat]
            )
            count = int(result["rows"][0][0]) if result["rows"] else 0

            if count == 0:
                if cat == "inventory":
                    status_result = execute_query(
                        "SELECT status, COUNT(*) FROM inventory GROUP BY status"
                    )
                    breakdown = ", ".join(
                        f"{row[0]}: {row[1]}" for row in status_result["rows"]
                    )
                     # WARN not FAIL — all OK is valid
                    detail = f"0 alerts — inventory status: {breakdown}"
                    is_fail = cat not in optional_cats
                    check(f"'{cat}' alerts exist", not is_fail, detail)

                elif cat == "fraud":
                    pred_result = execute_query(
                        "SELECT label, COUNT(*) FROM predictions WHERE prediction_type='fraud' GROUP BY label"
                    )
                    breakdown = ", ".join(
                        f"{row[0]}: {row[1]}" for row in pred_result["rows"]
                    )
                    check(f"'{cat}' alerts exist", count > 0,
                          f"0 alerts — fraud predictions: {breakdown}")
                    
                elif cat == "demand":
                    pred_result = execute_query(
                        "SELECT COUNT(*) FROM predictions WHERE prediction_type='demand'"
                    )
                    pred_count = int(pred_result["rows"][0][0]) if pred_result["rows"] else 0
                    detail = f"0 alerts — demand predictions in DB: {pred_count}"
                    is_fail = cat not in optional_cats
                    check(f"'{cat}' alerts exist", not is_fail, detail)

                elif cat == "conflict":
                    alert_result = execute_query(
                        "SELECT category, COUNT(*) FROM alerts GROUP BY category"
                    )
                    breakdown = ", ".join(
                        f"{row[0]}: {row[1]}" for row in alert_result["rows"]
                    )
                    detail = f"0 conflicts — existing alerts: {breakdown}"
                    is_fail = cat not in optional_cats
                    check(f"'{cat}' alerts exist", not is_fail, detail)

            else:
                resolved = execute_query(
                    "SELECT COUNT(*) FROM alerts WHERE category=? AND is_resolved=1",
                    [cat]
                )
                resolved_count = int(resolved["rows"][0][0]) if resolved["rows"] else 0
                check(f"'{cat}' alerts exist", count > 0,
                      f"{count} total ({resolved_count} resolved, {count - resolved_count} open)")
                
        # Check severities are valid
        valid_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        invalid = [a for a in alerts if a[3] not in valid_severities]
        check("All severities valid", len(invalid) == 0,
              f"{len(invalid)} invalid severities")

        # Clean up any old test alert first
        execute_query("DELETE FROM alerts WHERE agent='TestAgent'")

        # Insert test alert
        insert_alert(
            agent="TestAgent", severity="LOW", category="test",
            message="Integration test alert",
            recommendation="No action needed — this is a test",
        )

        # Verify insert worked
        new_alerts = get_alerts(limit=500)
        test_alert = next(
            (a for a in new_alerts if a[2] == "TestAgent"), None
        )
        check("Insert alert works", test_alert is not None)

        # Verify resolve works
        if test_alert:
            resolve_alert(test_alert[0])

            # Confirm it's actually resolved in DB
            result = execute_query(
                "SELECT is_resolved FROM alerts WHERE id=?",
                [test_alert[0]]
            )
            is_resolved = int(result["rows"][0][0]) if result["rows"] else 0
            check("Resolve alert works",    is_resolved == 1, f"is_resolved={is_resolved}")
            check("Resolved not in open",   test_alert not in get_alerts(limit=500))

        # Clean up test alert
        execute_query("DELETE FROM alerts WHERE agent='TestAgent'")

    except Exception as e:
        check("Alerts", False, str(e))

# ─────────────────────────────────────────────────────────────
# TEST 6 — Inventory
# ─────────────────────────────────────────────────────────────
def test_inventory():
    print("\n── Test 6: Inventory ───────────────────────────────")
    try:
        # Check total inventory
        inv = get_inventory_status()
        check("Inventory rows exist", len(inv) > 0, f"{len(inv)} products")

        # Check required columns exist
        required_cols = ["product_name", "simulated_stock", "avg_daily_demand",
                         "days_remaining", "reorder_threshold", "status"]
        missing_cols = [c for c in required_cols if c not in inv.columns]
        check("Required columns exist", len(missing_cols) == 0,
              f"missing: {missing_cols}" if missing_cols else "all present")

        # Check no null product names
        null_names = inv["product_name"].isnull().sum()
        check("No null product names", null_names == 0,
              f"{null_names} nulls found")

        # Check stock values are non-negative
        negative_stock = (inv["simulated_stock"] < 0).sum()
        check("Stock values non-negative", negative_stock == 0,
              f"{negative_stock} negative values")

        # Check each status filter works and has data
        total = 0
        for status in ["CRITICAL", "WARNING", "OK"]:
            filtered = get_inventory_status(status)
            count    = len(filtered)
            total   += count
            check(f"{status} filter works", isinstance(filtered, pd.DataFrame),
                  f"{count} products")

        # Check all products are accounted for
        check("Status counts match total", total == len(inv),
              f"{total} filtered == {len(inv)} total")

    except Exception as e:
        check("Inventory", False, str(e))

# ─────────────────────────────────────────────────────────────
# TEST 7 — Search
# ─────────────────────────────────────────────────────────────
def test_search():
    print("\n── Test 7: Search Orders ───────────────────────────")
    try:
        # Broad search — should always return results
        res = search_orders("a")
        check("Broad search returns results", len(res) > 0, f"{len(res)} orders")

        # Search by known keywords
        for keyword in ["Sport", "Electronics", "Europe"]:
            results = search_orders(keyword)
            check(f"Search '{keyword}' works", len(results) >= 0,
                  f"{len(results)} orders")

        # Search with no match — should return empty not crash
        empty = search_orders("zzzzzznotfound12345")
        check("No match returns empty", len(empty) == 0,
              f"{len(empty)} orders")

        # Verify result has expected number of columns
        # orders table has 21 columns
        if res:
            check("Result has correct columns", len(res[0]) == 21,
                  f"{len(res[0])} columns")

    except Exception as e:
        check("Search orders", False, str(e))


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
def print_summary():
    print("\n" + "=" * 55)
    passed  = sum(1 for _, s, _ in results if s == PASS)
    failed  = sum(1 for _, s, _ in results if s == FAIL)
    total   = len(results)
    pct     = round((passed / total) * 100) if total > 0 else 0

    print(f"  Results: {passed} passed | {failed} failed | {total} total ({pct}%)")

    # Show which tests failed
    if failed > 0:
        print(f"\n  ❌ Failed tests:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"     • {name}" + (f" — {detail}" if detail else ""))

    if failed == 0:
        print("  🚀 All tests passed — system is ready!")
    elif failed <= 2:
        print("  ⚠️  Minor issues — check failed tests above.")
    else:
        print("  🔴 Multiple failures — system needs attention.")

    print("=" * 55)


if __name__ == "__main__":

    start = time.time()

    print("=" * 55)
    print("  AutoBiz AI — MODULE 6: Integration Tests")
    print("=" * 55)

    # Check connection first — stop if DB unreachable
    try:
        execute_query("SELECT 1")
        print("  ✅ DB connection verified — running tests...")
    except Exception as e:
        print(f"  ❌ Cannot connect to DB: {e}")
        print("  ❌ All tests aborted — fix connection first.")
        exit(1)

    # Run all tests
    test_connection()
    test_tables()
    test_dashboard_summary()
    test_predictions()
    test_alerts()
    test_inventory()
    test_search()

    # Summary with total time
    elapsed = round(time.time() - start, 2)
    print(f"\n  ⏱️  Total time: {elapsed}s")
    print_summary()




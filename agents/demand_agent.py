# =============================================================================
# AGENT: DEMAND FORECASTING AGENT  —  agents/demand_agent.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from modules.database import execute_query, insert_alerts_batch

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEEKLY_DATA = os.path.join(BASE_DIR, "data", "processed", "weekly_demand.csv")

SURGE_THRESHOLD   = 1.25
DECLINE_THRESHOLD = 0.75


def _get_baselines():
    try:
        wd = pd.read_csv(WEEKLY_DATA)
        return wd.groupby("Product Name")["weekly_qty"].mean().round(1).to_dict()
    except Exception as e:
        print(f"  Warning: baseline load failed: {e}")
        return {}


def _get_forecasts():
    """
    Fetch next forecast week per product. Uses INNER JOIN subquery so
    predicted_value is guaranteed to match the MIN(label) week row.
    """
    try:
        result = execute_query("""
            SELECT p.product_name, p.label, p.predicted_value, p.confidence_score
            FROM predictions p
            INNER JOIN (
                SELECT product_name, MIN(label) AS min_week
                FROM predictions
                WHERE prediction_type = 'demand'
                GROUP BY product_name
            ) sub ON p.product_name = sub.product_name
                  AND p.label        = sub.min_week
            WHERE p.prediction_type = 'demand'
            ORDER BY p.product_name
        """)
        return [
            {"product": r[0], "week": r[1],
             "forecast": float(r[2] or 0), "ci": float(r[3] or 0)}
            for r in result["rows"]
        ]
    except Exception as e:
        print(f"  ERROR reading forecasts: {e}")
        return []


def _build_alert(product, forecast, baseline, direction, week, ci):
    pct    = abs((forecast - baseline) / baseline * 100) if baseline > 0 else 0
    ci_str = (f" (range: {max(0,forecast-ci/2):.0f}–{forecast+ci/2:.0f} units)"
              if ci > 0 else "")

    if direction == "surge":
        msg = (f"DEMAND SURGE — '{product}': Forecast week {week}: "
               f"{forecast:.0f} units{ci_str}. "
               f"{pct:.0f}% ABOVE historical avg of {baseline:.0f} units/week. "
               f"Demand spike expected — current stock may be insufficient.")
        rec = (f"INCREASE STOCK for '{product}': Procure ~{max(0,int(forecast-baseline))} "
               f"extra units above standard reorder. Alert procurement now.")
        return {"agent":"DemandAgent","severity":"HIGH","category":"demand",
                "message":msg,"recommendation":rec,"product_name":product}
    else:
        msg = (f"DEMAND DECLINE — '{product}': Forecast week {week}: "
               f"{forecast:.0f} units{ci_str}. "
               f"{pct:.0f}% BELOW historical avg of {baseline:.0f} units/week. "
               f"Risk of excess inventory if replenishment not adjusted.")
        rec = (f"DEFER REORDER for '{product}': Reduce next PO by ~"
               f"{max(0,int(baseline-forecast))} units. "
               f"Consider promotional pricing to stimulate demand.")
        return {"agent":"DemandAgent","severity":"MEDIUM","category":"demand",
                "message":msg,"recommendation":rec,"product_name":product}


def run_demand_agent():
    print("\n" + "=" * 55)
    print("  DemandAgent — Running")
    print("=" * 55)

    baselines = _get_baselines()
    forecasts = _get_forecasts()

    if not forecasts:
        print("  No demand forecasts in DB. Run machinelearning.py first.")
        return

    print(f"  Products with forecasts: {len(forecasts)}")
    print(f"  Products with baselines: {len(baselines)}")

    alerts = []
    surge_n = decline_n = normal_n = skip_n = 0

    for fc in forecasts:
        product  = fc["product"]
        forecast = fc["forecast"]
        week     = fc["week"]
        ci       = fc["ci"]
        baseline = baselines.get(product, 0)

        if baseline <= 0:
            skip_n += 1
            continue

        ratio = forecast / baseline
        if ratio >= SURGE_THRESHOLD:
            alerts.append(_build_alert(product, forecast, baseline, "surge", week, ci))
            surge_n += 1
        elif ratio <= DECLINE_THRESHOLD:
            alerts.append(_build_alert(product, forecast, baseline, "decline", week, ci))
            decline_n += 1
        else:
            normal_n += 1

    if alerts:
        print(f"  Saving {len(alerts)} alerts (batched)...")
        insert_alerts_batch(alerts, chunk_size=50)

    print(f"\n  Alerts saved:")
    print(f"    HIGH   — Surge  : {surge_n}")
    print(f"    MEDIUM — Decline: {decline_n}")
    print(f"    Normal (no alert): {normal_n}")
    print(f"    No baseline: {skip_n}")
    print(f"  DemandAgent complete.")


if __name__ == "__main__":
    run_demand_agent()

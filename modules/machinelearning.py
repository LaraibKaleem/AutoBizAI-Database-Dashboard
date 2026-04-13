# =============================================================================
# MODULE 2 — MACHINE LEARNING MODELS
# File: modules/machinelearning.py
#
# Three models, each solving a real business problem from this dataset:
#
#   MODEL 1 — LATE DELIVERY RISK (RandomForest classifier)
#     Target: will this order be delivered late? (Late_delivery_risk 0/1)
#     Why: 54.8% of orders arrive late. Shipping Mode and scheduled
#     lead time are the dominant signals. Accuracy: ~70%.
#     Business use: flag high-risk orders before they ship.
#
#   MODEL 2 — DEMAND FORECAST (Prophet time-series per product)
#     Target: how many units will sell next 8 weeks per product?
#     Why: 118 products, up to 147 weeks of history. Prophet detects
#     yearly seasonality (Jan peak, Nov low) reliably.
#     Business use: inventory replenishment planning.
#
#   MODEL 3 — INVENTORY RISK (XGBoost multi-class)
#     Target: is stock level SAFE / LOW / CRITICAL?
#     Business use: trigger reorder alerts.
#
# NOTE on Fraud: "Fraud" = Order Status == SUSPECTED_FRAUD.
#   Numeric features are statistically identical between fraud and legit
#   orders in this dataset — there is no learnable numeric pattern.
#   Fraud orders are stored in DB and used by the FraudAgent directly.
#
# Run: python modules/machinelearning.py
# =============================================================================

import os, sys, time, warnings, logging
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.database import execute_query, insert_predictions_batch

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA     = os.path.join(BASE_DIR, "data", "processed", "clean_data.csv")
INVENTORY_DATA = os.path.join(BASE_DIR, "data", "processed", "inventory_table.csv")
WEEKLY_DATA    = os.path.join(BASE_DIR, "data", "processed", "weekly_demand.csv")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# =============================================================================
# UTILITIES
# =============================================================================

def _model_ok(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return False
    try:
        joblib.load(path)
        return True
    except Exception:
        return False


def _preds_exist(ptype):
    try:
        r = execute_query(
            "SELECT COUNT(*) FROM predictions WHERE prediction_type=?", [ptype]
        )
        return int(r["rows"][0][0]) > 0
    except Exception:
        return False


def _insert_chunked(rows, chunk_size=200, label="predictions"):
    """
    Insert prediction rows in small chunks to avoid HTTP timeouts.
    chunk_size=200 keeps each request ~20KB — well within Turso free tier limits.
    """
    import time as _time
    total   = len(rows)
    saved   = 0
    failed  = 0

    for i in range(0, total, chunk_size):
        chunk = rows[i:i + chunk_size]
        try:
            insert_predictions_batch(chunk)
            saved += len(chunk)
        except Exception as e:
            failed += len(chunk)
            print(f"\n  WARNING: chunk {i//chunk_size + 1} failed ({e}), continuing...")
        if (i // chunk_size) % 10 == 0:
            print(f"  {label}: {min(saved, total):,}/{total:,} saved...", end="\r")
        _time.sleep(0.03)   # small pause between chunks

    if failed:
        print(f"\n  Saved {saved:,}/{total:,} {label} ({failed:,} failed).")
    else:
        print(f"\n  Saved {saved:,} {label}.")


# =============================================================================
# MODEL 1 — LATE DELIVERY RISK
# =============================================================================

DELIVERY_FEATURES = [
    "shipping_mode_enc",
    "Days for shipment (scheduled)",
    "market_enc",
    "customer_segment_enc",
    "department_enc",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount Rate",
    "order_month",
    "order_dayofweek",
]

# Human-readable names for XAI explanations
DELIVERY_FEATURE_LABELS = {
    "shipping_mode_enc"            : "Shipping Mode",
    "Days for shipment (scheduled)": "Scheduled Lead Time (days)",
    "market_enc"                   : "Sales Market",
    "customer_segment_enc"         : "Customer Segment",
    "department_enc"               : "Product Department",
    "Order Item Quantity"          : "Order Quantity",
    "Sales"                        : "Order Value ($)",
    "Order Item Discount Rate"     : "Discount Rate",
    "order_month"                  : "Order Month",
    "order_dayofweek"              : "Day of Week",
}


def train_delivery_model():
    print("\n" + "-" * 55)
    print("  MODEL 1 — Late Delivery Risk (RandomForest)")
    print("-" * 55)

    if _model_ok("delivery_model.pkl") and _preds_exist("delivery"):
        print("  Skipping — model and predictions already exist.")
        return joblib.load(os.path.join(MODELS_DIR, "delivery_model.pkl"))

    df = pd.read_csv(CLEAN_DATA)
    missing = [f for f in DELIVERY_FEATURES if f not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns in clean_data.csv: {missing}")
        print("  Re-run preprocessing.py first.")
        return None

    df_m = df[DELIVERY_FEATURES + ["Late_delivery_risk", "Order Id"]].dropna()
    X = df_m[DELIVERY_FEATURES]
    y = df_m["Late_delivery_risk"]

    print(f"  Rows: {len(X):,}  |  Late delivery rate: {y.mean()*100:.1f}%")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    print(f"  Trained in {time.time()-t0:.1f}s")

    y_pred = model.predict(X_te)
    print(f"  Test accuracy: {accuracy_score(y_te, y_pred)*100:.1f}%")
    print(classification_report(
        y_te, y_pred,
        target_names=["On Time / Cancelled", "Late Delivery"],
        zero_division=0
    ))

    importances = pd.Series(model.feature_importances_, index=DELIVERY_FEATURES)
    print("  Top drivers:")
    for feat, imp in importances.nlargest(4).items():
        print(f"    {DELIVERY_FEATURE_LABELS.get(feat, feat):40s} {imp*100:.1f}%")

    joblib.dump(model, os.path.join(MODELS_DIR, "delivery_model.pkl"))
    print("  Saved: delivery_model.pkl")

    # Persist high-risk delivery predictions to DB
    # Cap at top 5,000 highest-confidence late predictions to avoid DB flooding.
    # (~99K rows at prob>=0.60 would take 8+ minutes to insert via HTTP.)
    if not _preds_exist("delivery"):
        probs     = model.predict_proba(X)[:, 1]
        order_ids = df_m["Order Id"].values

        # Build (prob, index) pairs, keep only prob >= 0.60, sort descending
        high_risk = sorted(
            [(probs[i], i) for i in range(len(probs)) if probs[i] >= 0.60],
            key=lambda x: x[0], reverse=True
        )[:5000]   # top 5,000 highest-confidence late predictions

        rows = [
            {
                "prediction_type" : "delivery",
                "order_id"        : int(order_ids[idx]),
                "predicted_value" : round(float(prob), 4),
                "confidence_score": round(float(prob), 4),
                "label"           : "late",
                "product_name"    : None,
            }
            for prob, idx in high_risk
        ]
        print(f"  Saving {len(rows):,} highest-risk delivery predictions to DB...")
        _insert_chunked(rows, chunk_size=200, label="delivery predictions")

    return model


def predict_delivery_risk(order_data: dict):
    """
    Predict late delivery risk for one order.
    Returns (label, probability, top_drivers_dict)
    """
    model     = joblib.load(os.path.join(MODELS_DIR, "delivery_model.pkl"))
    row       = {f: order_data.get(f, 0) for f in DELIVERY_FEATURES}
    X         = pd.DataFrame([row])[DELIVERY_FEATURES]
    prob      = float(model.predict_proba(X)[0][1])
    label     = "late" if prob >= 0.6 else "on_time"
    importances = dict(zip(DELIVERY_FEATURES, model.feature_importances_))
    top3      = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
    drivers   = {DELIVERY_FEATURE_LABELS.get(f, f): round(v * 100, 1) for f, v in top3}
    return label, prob, drivers


# =============================================================================
# MODEL 2 — DEMAND FORECAST (Prophet per product + global)
# =============================================================================

def train_demand_model():
    """
    One Prophet model per product (min 20 weeks of history).
    Stores 8-week forecasts in DB as prediction_type='demand'.
    Also saves a global aggregate model as demand_model.pkl for the dashboard.
    Runtime: ~0.2s/product × 118 products = ~25 seconds.
    """
    print("\n" + "-" * 55)
    print("  MODEL 2 — Demand Forecast (Prophet per product)")
    print("-" * 55)

    try:
        from prophet import Prophet
    except ImportError:
        print("  ERROR: prophet not installed. Run: pip install prophet")
        return

    # Per-product forecasts
    if not _preds_exist("demand"):
        weekly = pd.read_csv(WEEKLY_DATA)
        weekly["ds"] = pd.to_datetime(weekly["ds"], errors="coerce")
        weekly = weekly.dropna(subset=["ds"])
        products = weekly["Product Name"].unique()
        trained, skipped, all_rows = 0, 0, []
        t0 = time.time()

        for product in products:
            pdf = (
                weekly[weekly["Product Name"] == product][["ds", "weekly_qty"]]
                .rename(columns={"weekly_qty": "y"})
                .dropna().sort_values("ds")
            )
            if len(pdf) < 20:
                skipped += 1
                continue
            try:
                m = Prophet(
                    yearly_seasonality="auto",
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode="additive",
                    changepoint_prior_scale=0.1,
                )
                m.fit(pdf)
                future   = m.make_future_dataframe(periods=8, freq="W")
                forecast = m.predict(future).tail(8)
                for _, r in forecast.iterrows():
                    yhat = max(float(r["yhat"]), 0)
                    ci   = max(float(r["yhat_upper"]) - float(r["yhat_lower"]), 0)
                    all_rows.append({
                        "prediction_type" : "demand",
                        "product_name"    : product,
                        "predicted_value" : round(yhat, 1),
                        "confidence_score": round(ci, 1),
                        "label"           : str(r["ds"].date()),
                        "order_id"        : None,
                    })
                trained += 1
            except Exception:
                skipped += 1

            if trained % 20 == 0 and trained > 0:
                print(f"  Forecasted {trained} products...", end="\r")

        elapsed = time.time() - t0
        print(f"  Forecasted {trained} products, skipped {skipped}  |  {elapsed:.0f}s")
        if all_rows:
            print(f"  Saving {len(all_rows):,} demand forecast rows to DB...")
            _insert_chunked(all_rows, chunk_size=200, label="demand predictions")
    else:
        print("  Per-product demand predictions already in DB — skipping.")

    # Global aggregate model (dashboard summary chart)
    if not _model_ok("demand_model.pkl"):
        weekly = pd.read_csv(WEEKLY_DATA)
        weekly["ds"] = pd.to_datetime(weekly["ds"], errors="coerce")
        total = weekly.groupby("ds")["weekly_qty"].sum().reset_index()
        total.columns = ["ds", "y"]
        total = total.dropna().sort_values("ds")
        try:
            m_global = Prophet(
                yearly_seasonality="auto", weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode="additive"
            )
            m_global.fit(total)
            joblib.dump(m_global, os.path.join(MODELS_DIR, "demand_model.pkl"))
            print("  Saved: demand_model.pkl (global aggregate model)")
        except Exception as e:
            print(f"  Warning: global model failed: {e}")
    else:
        print("  Global demand model already saved — skipping.")


def get_demand_forecast(product_name=None, weeks=8):
    """Fetch demand forecast from DB or global model."""
    if product_name:
        r = execute_query(
            "SELECT product_name, label, predicted_value, confidence_score "
            "FROM predictions WHERE prediction_type='demand' AND product_name=? "
            "ORDER BY label ASC LIMIT ?",
            [product_name, weeks]
        )
        if not r["rows"]:
            return pd.DataFrame()
        return pd.DataFrame(
            r["rows"],
            columns=["product_name", "week", "forecast_qty", "confidence_interval"]
        )

    model_path = os.path.join(MODELS_DIR, "demand_model.pkl")
    if not os.path.exists(model_path):
        return pd.DataFrame()
    try:
        from prophet import Prophet
        m = joblib.load(model_path)
        future = m.make_future_dataframe(periods=weeks, freq="W")
        fc = m.predict(future).tail(weeks)
        fc["yhat"] = fc["yhat"].clip(lower=0)
        return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
            columns={"ds": "week", "yhat": "forecast_qty"}
        )
    except Exception:
        return pd.DataFrame()


# =============================================================================
# MODEL 3 — INVENTORY RISK (XGBoost)
# =============================================================================

INVENTORY_FEATURES = [
    "simulated_stock", "avg_daily_demand", "days_remaining",
    "reorder_threshold", "total_sold", "avg_per_order", "product_price"
]
INVENTORY_LABELS    = {0: "safe", 1: "low", 2: "critical"}
INVENTORY_STATUS_MAP = {
    "OK": 0, "ok": 0, "safe": 0,
    "WARNING": 1, "Warning": 1, "warning": 1, "low": 1,
    "CRITICAL": 2, "Critical": 2, "critical": 2,
}


def train_inventory_model():
    print("\n" + "-" * 55)
    print("  MODEL 3 — Inventory Risk (XGBoost)")
    print("-" * 55)

    if _model_ok("inventory_model.pkl") and _preds_exist("inventory"):
        print("  Skipping — model and predictions already exist.")
        return joblib.load(os.path.join(MODELS_DIR, "inventory_model.pkl"))

    df = pd.read_csv(INVENTORY_DATA)
    print(f"  Products: {len(df)}  |  Status: {df['status'].value_counts().to_dict()}")

    df["status_enc"] = df["status"].map(INVENTORY_STATUS_MAP).fillna(0).astype(int)
    df_m = df[INVENTORY_FEATURES + ["status_enc"]].fillna(0)
    X, y = df_m[INVENTORY_FEATURES], df_m["status_enc"]

    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        objective="multi:softmax", num_class=3,
        random_state=42, eval_metric="mlogloss", verbosity=0
    )

    if len(X) >= 10:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_tr, y_tr)
        if len(y_te.unique()) > 1:
            print(f"  Accuracy: {accuracy_score(y_te, model.predict(X_te))*100:.1f}%")
    else:
        model.fit(X, y)

    joblib.dump(model, os.path.join(MODELS_DIR, "inventory_model.pkl"))
    print("  Saved: inventory_model.pkl")

    if not _preds_exist("inventory"):
        codes = model.predict(df[INVENTORY_FEATURES].fillna(0)).tolist()
        rows  = [
            {
                "prediction_type" : "inventory",
                "product_name"    : df.iloc[i]["Product Name"],
                "predicted_value" : float(codes[i]),
                "confidence_score": None,
                "order_id"        : None,
                "label"           : INVENTORY_LABELS[codes[i]],
            }
            for i in range(len(df))
        ]
        dist = {INVENTORY_LABELS[c]: codes.count(c) for c in [0, 1, 2]}
        print(f"  Saving {len(rows)} inventory predictions. Distribution: {dist}")
        _insert_chunked(rows, chunk_size=200, label="inventory predictions")

    return model


def predict_inventory_risk(product_data: dict):
    model = joblib.load(os.path.join(MODELS_DIR, "inventory_model.pkl"))
    row   = pd.DataFrame([product_data])[INVENTORY_FEATURES].fillna(0)
    code  = int(model.predict(row)[0])
    return INVENTORY_LABELS[code], code


# =============================================================================
# MAIN
# =============================================================================

def run_all_models():
    print("\n" + "#" * 55)
    print("  AutoBiz AI — MODULE 2: Machine Learning")
    print("#" * 55)

    try:
        execute_query("SELECT 1")
        print("  DB: Connected")
    except Exception as e:
        print(f"  ERROR: DB unreachable: {e}")
        return

    if not os.path.exists(CLEAN_DATA):
        print(f"  ERROR: {CLEAN_DATA} not found. Run preprocessing.py first.")
        return

    df = pd.read_csv(CLEAN_DATA)
    print(f"  Dataset: {len(df):,} rows")

    train_delivery_model()
    train_demand_model()
    train_inventory_model()

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    for fname in ["delivery_model.pkl", "demand_model.pkl", "inventory_model.pkl"]:
        path = os.path.join(MODELS_DIR, fname)
        status = f"{os.path.getsize(path)//1024} KB" if os.path.exists(path) else "MISSING"
        print(f"  {'OK' if os.path.exists(path) else 'X'}  {fname:35s} {status}")

    for ptype in ["delivery", "demand", "inventory"]:
        try:
            r = execute_query(
                "SELECT COUNT(*) FROM predictions WHERE prediction_type=?", [ptype]
            )
            print(f"  DB predictions [{ptype:12s}]: {int(r['rows'][0][0]):,}")
        except Exception:
            print(f"  DB predictions [{ptype:12s}]: ERROR")

    print("\n  Next step: python agents/run_all_agents.py")
    print("#" * 55)


if __name__ == "__main__":
    run_all_models()

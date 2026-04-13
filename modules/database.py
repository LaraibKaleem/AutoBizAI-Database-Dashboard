# =============================================================================
# MODULE 3 — DATABASE & DATA MANAGEMENT
# File: modules/database.py
# DB:   autobizai-db — Turso (Mumbai)
# =============================================================================

import os
import httpx
import pandas as pd
import certifi
from datetime import datetime
from dotenv import load_dotenv

os.environ['SSL_CERT_FILE']        = certifi.where()
os.environ['REQUESTS_CA_BUNDLE']   = certifi.where()

load_dotenv()

TURSO_URL   = os.getenv("TURSO_URL")
TURSO_TOKEN = os.getenv("TURSO_TOKEN")

print("URL:", os.getenv("TURSO_URL"))

if not TURSO_URL or not TURSO_TOKEN:
    raise ValueError("TURSO_URL or TURSO_TOKEN not found. Check your .env file.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

# ─── PERSISTENT HTTP CLIENT ────────────────────────────────────────────────────
# Using a module-level httpx.Client instead of httpx.post() on every call.
# This keeps a single persistent TCP connection with keep-alive, dramatically
# reducing SSL handshakes and preventing WinError 10053 / SSL failures that
# occur when Windows limits rapid new-connection creation.
_http_client: httpx.Client | None = None

def _get_client() -> httpx.Client:
    """Return (or lazily create) the persistent HTTP client."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.Client(
            verify   = certifi.where(),
            timeout  = httpx.Timeout(45.0, connect=10.0),
            limits   = httpx.Limits(max_keepalive_connections=5,
                                    max_connections=10,
                                    keepalive_expiry=30.0),
        )
    return _http_client


def close_client():
    """Call this on clean shutdown if needed."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        _http_client.close()
        _http_client = None


def execute_query(sql, args=None):
    """Single SQL via persistent HTTP client. Retries 3x on network errors."""
    import time as _t
    url   = TURSO_URL   or os.getenv("TURSO_URL") or ""
    token = TURSO_TOKEN or os.getenv("TURSO_TOKEN") or ""
    http_url = url.replace("libsql://", "https://") + "/v2/pipeline"
    headers  = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    if args:
        stmt = {"type":"execute","stmt":{"sql":sql,"args":[_to_turso_value(a) for a in args]}}
    else:
        stmt = {"type":"execute","stmt":{"sql":sql}}
    body = {"requests":[stmt,{"type":"close"}]}

    last_exc = None
    for attempt in range(3):
        try:
            response = _get_client().post(http_url, headers=headers, json=body)
            response.raise_for_status()
            data   = response.json()
            result = data["results"][0]
            if result["type"] == "error":
                raise Exception(f"Turso error: {result['error']}")
            rd   = result["response"]["result"]
            cols = [c["name"] for c in rd["cols"]]
            rows = [[_from_turso_value(v) for v in row] for row in rd["rows"]]
            return {"cols": cols, "rows": rows}
        except (httpx.ReadTimeout, httpx.ConnectTimeout,
                httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_exc = e
            close_client()
            _t.sleep((attempt+1)*2)
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                last_exc = e; close_client(); _t.sleep((attempt+1)*2)
            else:
                raise

    raise RuntimeError(f"execute_query failed after 3 attempts: {last_exc}") from last_exc


def execute_many(statements, _max_retries=3):
    """Batch SQL via persistent HTTP client. Retries 3x on network errors."""
    import time as _t

    url   = TURSO_URL   or os.getenv("TURSO_URL") or ""
    token = TURSO_TOKEN or os.getenv("TURSO_TOKEN") or ""
    http_url = url.replace("libsql://", "https://") + "/v2/pipeline"
    headers  = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    reqs = []
    for sql, args in statements:
        if args:
            reqs.append({"type":"execute","stmt":{"sql":sql,"args":[_to_turso_value(a) for a in args]}})
        else:
            reqs.append({"type":"execute","stmt":{"sql":sql}})
    reqs.append({"type":"close"})
    body = {"requests": reqs}

    last_exc = None
    for attempt in range(1, _max_retries + 1):
        try:
            response = _get_client().post(http_url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()
        except (httpx.ReadTimeout, httpx.ConnectTimeout,
                httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_exc = e
            close_client()
            wait = attempt * 2
            print(f"\n  Retry {attempt}/{_max_retries} in {wait}s...")
            _t.sleep(wait)
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                last_exc = e; close_client(); _t.sleep(attempt * 2)
            else:
                raise

    raise RuntimeError(f"execute_many failed after {_max_retries} attempts: {last_exc}") from last_exc


def _to_turso_value(v):
    """Convert Python value to Turso API format."""
    if v is None:
        return {"type": "null"}
    elif isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    elif isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    elif isinstance(v, float):
        import math
        if math.isnan(v):
            return {"type": "null"}
        return {"type": "float", "value": v}
    else:
        return {"type": "text", "value": str(v)}


def _from_turso_value(v):
    """Convert Turso API value back to Python."""
    if v["type"] == "null":
        return None
    elif v["type"] == "integer":
        return int(v["value"])
    elif v["type"] == "float":
        return float(v["value"])
    else:
        return v["value"]


# ─── CREATE TABLES ────────────────────────────────────────────────────────────
def create_tables():
    statements = [
        ("CREATE TABLE IF NOT EXISTS orders (order_id INTEGER, order_date TEXT, product_name TEXT, category TEXT, department TEXT, market TEXT, order_region TEXT, customer_segment TEXT, sales REAL, quantity INTEGER, profit REAL, discount_rate REAL, shipping_mode TEXT, delivery_status TEXT, delay_days REAL, fraud_label INTEGER, is_high_value INTEGER, order_year INTEGER, order_month INTEGER, order_week INTEGER, order_season TEXT)", []),
        ("CREATE TABLE IF NOT EXISTS inventory (product_name TEXT PRIMARY KEY, total_sold INTEGER, "
        "total_orders INTEGER, avg_per_order REAL, product_price REAL, category TEXT, "
        "department TEXT, simulated_stock INTEGER, reorder_threshold REAL, avg_daily_demand REAL, "
        "days_remaining REAL, status TEXT)", []),
        ("CREATE TABLE IF NOT EXISTS weekly_demand (product_name TEXT, year INTEGER, week INTEGER, total_quantity REAL, total_sales REAL)", []),
        ("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, created_at TEXT, prediction_type TEXT, product_name TEXT, order_id INTEGER, predicted_value REAL, confidence_score REAL, label TEXT)", []),
        ("CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY, created_at TEXT, agent TEXT, severity TEXT, category TEXT, product_name TEXT, order_id INTEGER, message TEXT, recommendation TEXT, is_resolved INTEGER DEFAULT 0)", []),
    ]
    execute_many(statements)
    print("✅ All 5 tables created in autobizai-db (Mumbai)")


def load_processed_data():
    import time

    # ── Orders ────────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(PROC_DIR, "clean_data.csv"))

    # IMPORTANT: ML models read clean_data.csv directly from disk (full dataset).
    # The DB orders table is only used for: dashboard display, fraud table, search.
    # Loading 180K rows into a free-tier cloud DB via HTTP takes 10+ minutes.
    # Fix: load ALL fraud rows + a 6K non-fraud sample = ~10K rows total.
    # This loads in ~30 seconds and covers all dashboard use cases.
    fraud_rows     = df[df["fraud_label"] == 1]
    non_fraud_rows = df[df["fraud_label"] == 0]
    non_fraud_cap  = min(6000, len(non_fraud_rows))
    non_fraud_sample = non_fraud_rows.sample(n=non_fraud_cap, random_state=42)
    df_db = pd.concat([fraud_rows, non_fraud_sample], ignore_index=True)

    df_db = df_db[[
        "Order Id", "order date (DateOrders)", "Product Name",
        "Category Name", "Department Name", "Market", "Order Region",
        "Customer Segment", "Sales", "Order Item Quantity",
        "Order Profit Per Order", "Order Item Discount Rate",
        "Shipping Mode", "Delivery Status", "delay_days",
        "fraud_label", "is_high_value", "order_year",
        "order_month", "order_week", "order_season"
    ]]
    df_db.columns = [
        "order_id", "order_date", "product_name", "category",
        "department", "market", "order_region", "customer_segment",
        "sales", "quantity", "profit", "discount_rate",
        "shipping_mode", "delivery_status", "delay_days",
        "fraud_label", "is_high_value", "order_year",
        "order_month", "order_week", "order_season"
    ]

    print(f"  Loading {len(df_db):,} orders to DB "
          f"({len(fraud_rows):,} fraud + {non_fraud_cap:,} non-fraud sample)...")
    print(f"  Full dataset ({len(df):,} rows) stays on disk for ML training.")

    execute_query("DELETE FROM orders")

    BATCH = 100   # ~93 KB per request — safe for Turso free tier
    rows  = df_db.values.tolist()
    total = len(rows)
    failed_batches = 0

    for i in range(0, total, BATCH):
        batch = rows[i:i + BATCH]
        statements = [
            ("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            for row in batch
        ]
        try:
            execute_many(statements)
        except RuntimeError as e:
            failed_batches += 1
            print(f"\n  ❌ Batch {i//BATCH + 1} failed permanently: {e}")
            continue
        pct = min(i + BATCH, total)
        print(f"  Orders: {pct:,}/{total:,} rows loaded...", end="\r")
        time.sleep(0.05)   # small pause to avoid rate-limiting

    if failed_batches:
        print(f"\n  ⚠️  {failed_batches} batch(es) failed — some orders may be missing.")
    print(f"\n✅ Orders loaded: {total - failed_batches * BATCH:,} rows")

    # ── Inventory ─────────────────────────────────────────────────────────
    inv = pd.read_csv(os.path.join(PROC_DIR, "inventory_table.csv"))
    execute_query("DELETE FROM inventory")
    statements = [
         ("""
            INSERT OR REPLACE INTO inventory (
                product_name,
                total_sold,
                total_orders,
                avg_per_order,
                product_price,
                category,
                department,
                simulated_stock,
                reorder_threshold,
                avg_daily_demand,
                days_remaining,
                status
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, list(row))
      
        # ("INSERT OR REPLACE INTO inventory VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", list(row))
        for row in inv.values.tolist()
    ]
    execute_many(statements)
    print(f"✅ Inventory loaded: {len(inv):,} rows")

    # ── Weekly demand ──────────────────────────────────────────────────────
    wd = pd.read_csv(os.path.join(PROC_DIR, "weekly_demand.csv"))
    wd = wd[["Product Name", "order_year", "order_week",
             "weekly_qty", "weekly_sales"]]
    wd.columns = ["product_name", "year", "week",
                  "total_quantity", "total_sales"]
    execute_query("DELETE FROM weekly_demand")
    wd_rows = wd.values.tolist()
    for i in range(0, len(wd_rows), 200):
        batch = wd_rows[i:i + 200]
        statements = [
            ("INSERT INTO weekly_demand VALUES (?,?,?,?,?)", list(row))
            for row in batch
        ]
        execute_many(statements)
        time.sleep(0.05)
        print(f"  Weekly demand: {min(i+200, len(wd_rows)):,}/{len(wd_rows):,} rows...", end="\r")
    print(f"\n✅ Weekly demand loaded: {len(wd_rows):,} rows")



# ─── PREDICTION FUNCTIONS  ────────────────────────────────
def insert_prediction(prediction_type, product_name=None, order_id=None,
                      predicted_value=None, confidence_score=None, label=None):
    execute_query(
        "INSERT INTO predictions (created_at,prediction_type,product_name,order_id,predicted_value,confidence_score,label) VALUES (?,?,?,?,?,?,?)",
        [datetime.now().isoformat(), prediction_type, product_name,
         order_id, predicted_value, confidence_score, label]
    )

def insert_predictions_batch(rows, _chunk=150):
    """
    Insert prediction rows in chunks of 150 to stay within Turso free-tier
    payload limits (~14KB per chunk). Safe to call with any number of rows.
    """
    import time as _time
    for i in range(0, len(rows), _chunk):
        chunk = rows[i:i + _chunk]
        statements = [
            (
                "INSERT INTO predictions (created_at,prediction_type,product_name,"
                "order_id,predicted_value,confidence_score,label) VALUES (?,?,?,?,?,?,?)",
                [
                    datetime.now().isoformat(),
                    r.get("prediction_type"),
                    r.get("product_name"),
                    r.get("order_id"),
                    r.get("predicted_value"),
                    r.get("confidence_score"),
                    r.get("label"),
                ]
            )
            for r in chunk
        ]
        execute_many(statements)
        _time.sleep(0.02)

    #100
def get_predictions(prediction_type=None, limit=1000): 
    if prediction_type:
        result = execute_query(
            "SELECT * FROM predictions WHERE prediction_type=? ORDER BY created_at DESC LIMIT ?",
            [prediction_type, limit]
        )
    else:
        result = execute_query(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
            [limit]
        )
    return result["rows"]


# ─── ALERT FUNCTIONS (Agents call these) ─────────────────────────────────────

# def is_alert_duplicate(agent, category, message, product_name=None, order_id=None):
#     """
#     Check if exact same alert already exists in DB.
#     Returns True if duplicate found — skip insert.
#     """
#     try:
#         if order_id:
#             result = execute_query(
#                 "SELECT COUNT(*) FROM alerts WHERE agent=? AND category=? AND order_id=?",
#                 [agent, category, order_id]
#             )
#         elif product_name:
#             result = execute_query(
#                 "SELECT COUNT(*) FROM alerts WHERE agent=? AND category=? AND product_name=?",
#                 [agent, category, product_name]
#             )
#         else:
#             result = execute_query(
#                 "SELECT COUNT(*) FROM alerts WHERE agent=? AND category=? AND message=?",
#                 [agent, category, message]
#             )
#         rows  = result.get("rows", [])
#         count = int(rows[0][0]) if rows and rows[0][0] else 0
#         return count > 0
#     except Exception:
#         return False

def insert_alert(agent, severity, category, message, recommendation,
                 product_name=None, order_id=None):
    """Single alert insert with retry on ConnectError."""
    import time as _t
    for attempt in range(3):
        try:
            if order_id:
                execute_query(
                    "DELETE FROM alerts WHERE agent=? AND category=? AND order_id=?",
                    [agent, category, order_id]
                )
            elif product_name:
                execute_query(
                    "DELETE FROM alerts WHERE agent=? AND category=? AND product_name=?",
                    [agent, category, product_name]
                )
            else:
                execute_query(
                    "DELETE FROM alerts WHERE agent=? AND category=? AND message=?",
                    [agent, category, message]
                )
            execute_query(
                "INSERT INTO alerts (created_at,agent,severity,category,"
                "product_name,order_id,message,recommendation) VALUES (?,?,?,?,?,?,?,?)",
                [datetime.now().isoformat(), agent, severity, category,
                 product_name, order_id, message, recommendation]
            )
            return
        except Exception as e:
            if attempt < 2:
                _t.sleep(2)
            else:
                raise


def insert_alerts_batch(alerts_list, chunk_size=50):
    """
    Insert a list of alert dicts in batched execute_many calls.
    Each dict must have: agent, severity, category, message, recommendation
    Optional keys: product_name, order_id

    Strategy:
      1. DELETE all existing alerts for this agent+category in one call
      2. INSERT all new alerts in chunks of chunk_size

    This replaces the loop of individual insert_alert() calls and reduces
    HTTP requests from (N × 2) down to (1 + N/chunk_size) — critical for
    agents like FraudAgent that generate 500 alerts.
    """
    import time as _t
    if not alerts_list:
        return

    # Group by (agent, category) for efficient bulk delete
    agent_cats = {(a["agent"], a["category"]) for a in alerts_list}
    del_stmts  = [
        ("DELETE FROM alerts WHERE agent=? AND category=?", [ag, cat])
        for ag, cat in agent_cats
    ]
    try:
        execute_many(del_stmts)
    except Exception as e:
        print(f"  Warning: bulk delete failed ({e}), continuing with inserts...")

    # INSERT in chunks
    ts  = datetime.now().isoformat()
    rows = [
        (
            ts,
            a["agent"], a["severity"], a["category"],
            a.get("product_name"), a.get("order_id"),
            a["message"], a["recommendation"]
        )
        for a in alerts_list
    ]

    sql = ("INSERT INTO alerts (created_at,agent,severity,category,"
           "product_name,order_id,message,recommendation) VALUES (?,?,?,?,?,?,?,?)")

    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        stmts = [(sql, list(r)) for r in chunk]
        for attempt in range(3):
            try:
                execute_many(stmts)
                break
            except Exception as e:
                if attempt < 2:
                    _t.sleep(2 * (attempt + 1))
                else:
                    print(f"  Warning: alert chunk {i//chunk_size+1} failed: {e}")
        _t.sleep(0.5)   # give TCP stack breathing room between chunks

#50
def get_alerts(severity=None, resolved=False, limit=500):
    query  = "SELECT * FROM alerts WHERE 1=1"
    params = []
    if severity:
        query += " AND severity=?"
        params.append(severity)
    if not resolved:
        query += " AND is_resolved=0"
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    return execute_query(query, params)["rows"]


def resolve_alert(alert_id):
    execute_query(
        "UPDATE alerts SET is_resolved=1 WHERE id=?",
        [alert_id]
    )


# ─── DASHBOARD QUERY FUNCTIONS ────────────────────────────
def get_dashboard_summary():
    return {
        "total_orders"   : execute_query("SELECT COUNT(*) FROM orders")["rows"][0][0],
        "fraud_orders"   : execute_query("SELECT COUNT(*) FROM orders WHERE fraud_label=1")["rows"][0][0],
        "open_alerts"    : execute_query("SELECT COUNT(*) FROM alerts WHERE is_resolved=0")["rows"][0][0],
        "critical_stock" : execute_query("SELECT COUNT(*) FROM inventory WHERE status='CRITICAL'")["rows"][0][0],
    }


def get_inventory_status(status_filter=None):
    if status_filter:
        result = execute_query(
            "SELECT * FROM inventory WHERE status=?", [status_filter]
        )
    else:
        result = execute_query("SELECT * FROM inventory")
    # Column order MUST match the CREATE TABLE / INSERT order exactly:
    # product_name, total_sold, total_orders, avg_per_order, product_price,
    # category, department, simulated_stock, reorder_threshold,
    # avg_daily_demand, days_remaining, status
    cols = [
        "product_name", "total_sold", "total_orders", "avg_per_order",
        "product_price", "category", "department", "simulated_stock",
        "reorder_threshold", "avg_daily_demand", "days_remaining", "status"
    ]
    return pd.DataFrame(result["rows"], columns=cols)

#200
def get_fraud_orders(limit=5000):
    result = execute_query("""
        SELECT order_id, order_date, product_name, customer_segment,
               market, order_region, sales, delivery_status, delay_days
        FROM orders WHERE fraud_label=1
        ORDER BY sales DESC LIMIT ?
    """, [limit])
    cols = ["order_id","order_date","product_name","customer_segment",
            "market","order_region","sales","delivery_status","delay_days"]
    return pd.DataFrame(result["rows"], columns=cols)

def get_weekly_demand(product_name=None, limit=100):
    if product_name:
        result = execute_query(
            "SELECT * FROM weekly_demand WHERE product_name=? ORDER BY year DESC, week DESC LIMIT ?",
            [product_name, limit]
        )
    else:
        result = execute_query(
            "SELECT * FROM weekly_demand ORDER BY year DESC, week DESC LIMIT ?",
            [limit]
        )
    cols = ["product_name", "year", "week", "total_quantity", "total_sales"]
    return pd.DataFrame(result["rows"], columns=cols)

def search_orders(keyword):
    kw = f"%{keyword}%"
    result = execute_query(
        "SELECT * FROM orders WHERE product_name LIKE ? OR order_region LIKE ? OR category LIKE ? LIMIT 100",
        [kw, kw, kw]
    )
    return result["rows"]
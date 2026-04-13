# =============================================================================
# init_db.py
# Run ONCE to set up the entire database.
# Safe to run multiple times — will NOT duplicate data.
# If preprocessing was re-run, automatically detects stale data and reloads.
# =============================================================================

import pandas as pd
import os
from modules.database import create_tables, load_processed_data, execute_query


def is_connected():
    try:
        execute_query("SELECT 1")
        return True
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False


def is_tables_exist():
    try:
        execute_query("SELECT COUNT(*) FROM orders")
        return True
    except Exception:
        return False


def is_data_loaded():
    """
    Returns True only if DB orders count matches what load_processed_data()
    would insert: all fraud rows + min(6000, non-fraud rows).
    Forces reload if preprocessing was re-run and counts changed.
    """
    try:
        result   = execute_query("SELECT COUNT(*) FROM orders")
        rows     = result.get("rows", [])
        db_count = int(rows[0][0]) if rows and rows[0][0] else 0

        if db_count == 0:
            return False

        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "processed", "clean_data.csv"
        )
        df          = pd.read_csv(csv_path, usecols=["fraud_label"])
        fraud_n     = int((df["fraud_label"] == 1).sum())
        non_fraud_n = min(6000, int((df["fraud_label"] == 0).sum()))
        expected    = fraud_n + non_fraud_n

        # Allow +-5% tolerance for partial batch failures / rounding
        tolerance = max(50, int(expected * 0.05))
        if abs(db_count - expected) > tolerance:
            print(f"  Warning: DB has {db_count:,} orders, expected ~{expected:,} -- will reload.")
            return False

        return True
    except Exception:
        return False


if __name__ == "__main__":
    print("=" * 55)
    print("  AutoBiz AI -- MODULE 3: Database Initialisation")
    print("=" * 55)

    print("Connecting to autobizai-db (Turso - Mumbai)...")
    if not is_connected():
        print("  Cannot reach database. Check TURSO_URL and TURSO_TOKEN in .env")
        exit(1)
    print("  Connected to Turso successfully.")

    if is_tables_exist():
        print("  Tables already exist -- skipping create.")
    else:
        print("  Creating tables...")
        create_tables()

    if is_data_loaded():
        print("  Data already loaded and up to date -- skipping load.")
        print("  Next step: python modules/machinelearning.py")
    else:
        print("  Loading data now (orders capped at ~10K rows for speed)...")
        load_processed_data()
        print("\n  Database fully ready.")
        print("  Next step: python modules/machinelearning.py")

    print("=" * 55)

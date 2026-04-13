# =============================================================================
# MODULE 1 — DATA PREPROCESSING
# File: modules/preprocessing.py
#
# What this file does (plain English):
#   1. Loads the raw CSV dataset
#   2. Removes columns we don't need (personal info, useless fields)
#   3. Fixes data types (dates, numbers)
#   4. Creates new useful columns (delay_days, fraud_label, etc.)
#   5. Builds the inventory simulation table
#   6. Handles missing values
#   7. Encodes text columns into numbers (so ML models can read them)
#   8. Normalizes/scales numeric columns (so no one column dominates)
#   9. Saves everything to data/processed/ folder
#  10. Saves all encoders + scalers to models/ folder (for reuse in M2, M4)
#
# NOTE: stratified sampling has been removed. The full 180K-row dataset is used.
# See run_preprocessing() docstring for the full explanation.
#
# Run this file first before anything else.
# =============================================================================

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ── PATHS ─────────────────────────────────────────────────────────────────────
# os.path.dirname(__file__)  = the folder where this file lives  (modules/)
# ..  = go one folder up                                          (autobiz_ai/)
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA    = os.path.join(BASE_DIR, "data", "raw",       "DataCoSupplyChainDataset.csv")
LOGS_DATA   = os.path.join(BASE_DIR, "data", "raw",       "tokenized_access_logs.csv")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

# Make sure output folders exist
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# =============================================================================
# STEP 1 — LOAD RAW DATASET
# =============================================================================
def load_raw_data():
    """
    Load the CSV file.
    encoding='latin-1' is required because the file has special characters
    like accented letters (é, ñ) that UTF-8 can't read.
    """
    print("\n" + "="*60)
    print("STEP 1: Loading raw dataset...")
    print("="*60)

    df = pd.read_csv(RAW_DATA, encoding='latin-1')
    
    print(f"  Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")
    return df


# =============================================================================
# STEP 2 — DROP USELESS AND PRIVATE COLUMNS
# =============================================================================
def drop_columns(df):
    """
    Remove columns that are either:
      - Personal/private info (email, name, password) — not needed for ML
      - Completely empty (Product Description has 0 non-null values)
      - Not useful for any agent or model
    """
    print("\n" + "="*60)
    print("STEP 2: Dropping unnecessary columns...")
    print("="*60)

    columns_to_drop = [
        # ── Personal Information (privacy, not useful for ML) ──
        "Customer Email",
        "Customer Fname",
        "Customer Lname",
        "Customer Password",
        "Customer Street",
        "Customer Zipcode",
        "Customer City",
        "Customer Country",
        "Customer State",

        # ── Fully empty columns ──
        "Product Description",   # 100% null

        # ── Not useful for analysis ──
        "Product Image",         # just a URL
        "Latitude",
        "Longitude",
        "Order Zipcode",         # 86% missing
        "Order Item Id",         # just a row counter
        "Order Item Cardprod Id",# duplicate of Product Card Id
        "Order City",            # too granular, use Order Region instead
        "Order Country",         # redundant with Market
        "Order State",           # too granular
        "Order Customer Id",     # same as Customer Id
    ]

    before = df.shape[1]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    after = df.shape[1]

    print(f"  Dropped {before - after} columns. Remaining: {after} columns")
    return df


# =============================================================================
# STEP 3 — FIX DATA TYPES
# =============================================================================
def fix_datatypes(df):
    """
    Convert columns to the correct data type.
    Dates come in as strings — we parse them into real datetime objects.
    This lets us extract year, month, week, etc. in the next step.
    """
    print("\n" + "="*60)
    print("STEP 3: Fixing data types...")
    print("="*60)

    # Parse order date  (format examples: '1/31/2018 22:56', '1/13/2018 12:27')
    df['order date (DateOrders)'] = pd.to_datetime(
        df['order date (DateOrders)'],
        format='%m/%d/%Y %H:%M',
        errors='coerce'   # if parsing fails, put NaT (not a date) instead of crashing
    )

    # Parse shipping date
    df['shipping date (DateOrders)'] = pd.to_datetime(
        df['shipping date (DateOrders)'],
        format='%m/%d/%Y %H:%M',
        errors='coerce'
    )

    # Make sure numeric columns are actually numeric
    numeric_cols = [
        'Days for shipping (real)',
        'Days for shipment (scheduled)',
        'Order Item Quantity',
        'Order Item Discount Rate',
        'Order Item Profit Ratio',
        'Sales',
        'Order Profit Per Order',
        'Order Item Total',
        'Benefit per order',
        'Product Price',
        'Order Item Product Price',
        'Order Item Discount',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Integer columns
    int_cols = ['Late_delivery_risk', 'Product Status', 'Order Item Quantity']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    print(f"  Date columns parsed. Numeric types confirmed.")
    return df


# =============================================================================
# STEP 4 — HANDLE MISSING VALUES
# =============================================================================
def handle_missing_values(df):
    """
    Fill in or remove missing values so ML models don't crash.

    Rules:
      - Numeric columns  → fill with median (median is better than mean
                           because it's not affected by extreme outliers)
      - Text columns     → fill with 'Unknown'
      - Rows where date  → could not be parsed → drop them
    """
    print("\n" + "="*60)
    print("STEP 4: Handling missing values...")
    print("="*60)

    # Drop rows where order date could not be parsed (very rare)
    before = len(df)
    df = df.dropna(subset=['order date (DateOrders)'])
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with unparseable dates")

    # Fill numeric missing values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled {missing} missing values in '{col}' with median={median_val:.2f}")

    # Fill text missing values with 'Unknown'
    text_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in text_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col] = df[col].fillna('Unknown')
            print(f"  Filled {missing} missing text values in '{col}' with 'Unknown'")

    print(f"  Total missing values remaining: {df.isnull().sum().sum()}")
    return df


# =============================================================================
# STEP 5 — FEATURE ENGINEERING
# =============================================================================
def feature_engineering(df):
    """
    Create NEW columns from existing data.
    These new features give ML models more useful information.

    New columns created:
      order_year       → year the order was placed
      order_month      → month (1=Jan, 12=Dec)
      order_week       → week number of the year (1–52)
      order_dayofweek  → day of week (0=Monday, 6=Sunday)
      delay_days       → how many extra days shipping took vs scheduled
                         (negative = arrived early, positive = arrived late)
      fraud_label      → 1 if order was fraud, 0 if not
      is_late          → same as Late_delivery_risk (kept for clarity)
      profit_margin    → profit as percentage of sales
      discount_amount  → discount in dollars (not percentage)
    """
    print("\n" + "="*60)
    print("STEP 5: Feature engineering (creating new columns)...")
    print("="*60)

    # ── Date-based features ──────────────────────────────────────────────────
    df['order_year']      = df['order date (DateOrders)'].dt.year
    df['order_month']     = df['order date (DateOrders)'].dt.month
    df['order_week']      = df['order date (DateOrders)'].dt.isocalendar().week.astype(int)
    df['order_dayofweek'] = df['order date (DateOrders)'].dt.dayofweek
    # Season: 1=Winter(Dec-Feb), 2=Spring(Mar-May), 3=Summer(Jun-Aug), 4=Autumn(Sep-Nov)
    df['order_season']    = df['order_month'].map(
        {12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4}
    )

    # ── Shipping delay feature ───────────────────────────────────────────────
    # Positive = late, Negative = early, Zero = exactly on time
    df['delay_days'] = (
        df['Days for shipping (real)'] - df['Days for shipment (scheduled)']
    )

    # ── Fraud label ──────────────────────────────────────────────────────────
    # 1 = this order was fraud, 0 = normal order
    df['fraud_label'] = (df['Order Status'] == 'SUSPECTED_FRAUD').astype(int)

    # ── Profit margin ────────────────────────────────────────────────────────
    # What percentage of the sale was profit?
    # Avoid dividing by zero — use np.where
    df['profit_margin'] = np.where(
        df['Sales'] != 0,
        (df['Order Profit Per Order'] / df['Sales']) * 100,
        0
    )
    df['profit_margin'] = df['profit_margin'].round(4)

    # ── Discount amount in dollars ───────────────────────────────────────────
    df['discount_amount'] = df['Order Item Product Price'] * df['Order Item Discount Rate']
    df['discount_amount'] = df['discount_amount'].round(4)

    # ── Is high-value order ──────────────────────────────────────────────────
    sales_75th = df['Sales'].quantile(0.75)
    df['is_high_value'] = (df['Sales'] >= sales_75th).astype(int)

    print(f"  Created: order_year, order_month, order_week, order_dayofweek, order_season")
    print(f"  Created: delay_days  (range: {df['delay_days'].min()} to {df['delay_days'].max()})")
    print(f"  Created: fraud_label (fraud rows: {df['fraud_label'].sum():,})")
    print(f"  Created: profit_margin, discount_amount, is_high_value")

    return df


# =============================================================================
# STEP 6 — BUILD INVENTORY SIMULATION TABLE
# =============================================================================
def build_inventory_table(df):
    """
    The dataset has NO inventory/stock column.
    We SIMULATE it from historical order data.

    Formula:
      total_sold       = how many units of this product were ever ordered
      avg_per_order    = average units per single order
      simulated_stock  = total_sold × 0.15
                         (we assume 15% of all historical sales still remain as stock)
      reorder_threshold = avg_per_order × 7
                         (keep at least 7 days worth of average demand in stock)
      days_remaining   = simulated_stock ÷ avg_daily_demand
                         where avg_daily_demand = total_sold ÷ total_days_in_dataset
      status           = CRITICAL if days_remaining < 7
                         WARNING  if days_remaining < 14
                         OK       otherwise
    """
    print("\n" + "="*60)
    print("STEP 6: Building inventory simulation table...")
    print("="*60)

    # How many days does the dataset span?
    total_days = (df['order date (DateOrders)'].max() - df['order date (DateOrders)'].min()).days
    total_days = max(total_days, 1)  # avoid division by zero

    # Group by product — compute sales statistics
    inventory = df.groupby('Product Name').agg(
        total_sold    = ('Order Item Quantity', 'sum'),
        total_orders  = ('Order Item Quantity', 'count'),
        avg_per_order = ('Order Item Quantity', 'mean'),
        product_price = ('Product Price', 'first'),
        category      = ('Category Name', 'first'),
        department    = ('Department Name', 'first'),
    ).reset_index()

    # ── Apply simulation formulas ─────────────────────────────────────────────
    inventory['avg_daily_demand']  = (inventory['total_sold'] / total_days).round(4)
    inventory['reorder_threshold'] = (inventory['avg_per_order'] * 7).round(2)

    # FIX: the original formula was simulated_stock = total_sold * 0.15, which
    # made days_remaining = (total_sold*0.15) / (total_sold/total_days)
    #                      = 0.15 * total_days = ~168 days for EVERY product.
    # That means ALL products always show status=OK — completely useless.
    #
    # Correct approach: assign each product a realistic random "days of supply
    # on hand" (5–45 days), seeded so results are reproducible.
    # This mirrors real-world variance: some products are well-stocked,
    # others are nearly depleted.
    rng = np.random.default_rng(seed=42)
    days_of_supply = rng.integers(5, 46, size=len(inventory))  # 5 to 45 days
    inventory['simulated_stock'] = (
        inventory['avg_daily_demand'] * days_of_supply
    ).round(0).astype(int).clip(lower=1)   # at least 1 unit on hand

    # Days remaining before stockout
    inventory['days_remaining'] = np.where(
        inventory['avg_daily_demand'] > 0,
        (inventory['simulated_stock'] / inventory['avg_daily_demand']).round(1),
        999   # if avg_daily_demand = 0, stock lasts "forever"
    )

    # Assign status
    def get_status(days):
        if days < 7:  return 'CRITICAL'
        if days < 14: return 'WARNING'
        return 'OK'

    inventory['status'] = inventory['days_remaining'].apply(get_status)

    # Summary
    status_counts = inventory['status'].value_counts()
    print(f"  Products tracked: {len(inventory):,}")
    print(f"  CRITICAL (< 7 days): {status_counts.get('CRITICAL', 0)}")
    print(f"  WARNING  (< 14 days): {status_counts.get('WARNING', 0)}")
    print(f"  OK: {status_counts.get('OK', 0)}")

    # Save inventory table
    inventory_path = os.path.join(PROC_DIR, "inventory_table.csv")
    inventory.to_csv(inventory_path, index=False)
    print(f"  Saved: {inventory_path}")

    return inventory


# =============================================================================
# STEP 7 — SMART STRATIFIED SAMPLING
# =============================================================================
def stratified_sample(df, target_size=4000):
    """
    The full dataset has 180,519 rows — too large for a prototype.
    We take a smart sample that keeps the proportions of each group.

    Strategy:
      1. ALWAYS keep ALL 4,062 fraud rows (they are rare and critical)
      2. Sample remaining rows proportionally from each Order Status class
      3. Result: ~5,000–6,000 rows total (fraud rows push it over 4,000)

    Why not just random sample?
      A random sample might accidentally drop most fraud rows,
      which would make fraud detection impossible to train.
    """
    print("\n" + "="*60)
    print(f"STEP 7: Smart stratified sampling (target ~{target_size} rows)...")
    print("="*60)

    # 1. Separate fraud rows — keep ALL of them
    fraud_rows    = df[df['fraud_label'] == 1].copy()
    non_fraud_rows = df[df['fraud_label'] == 0].copy()

    print(f"  Fraud rows (keep all): {len(fraud_rows):,}")
    print(f"  Non-fraud rows (will sample from): {len(non_fraud_rows):,}")

    # 2. Sample from non-fraud rows proportionally by Order Status
    remaining_budget = target_size - len(fraud_rows)
    remaining_budget = max(remaining_budget, 1000)  # at least 1000 non-fraud rows

    # Calculate how many to take from each status
    status_counts  = non_fraud_rows['Order Status'].value_counts()
    total_non_fraud = len(non_fraud_rows)

    sampled_parts = []
    for status, count in status_counts.items():
        proportion = count / total_non_fraud
        n_to_take  = max(int(proportion * remaining_budget), 10)  # at least 10 per class
        n_to_take  = min(n_to_take, count)                         # can't take more than exists
        sampled    = non_fraud_rows[non_fraud_rows['Order Status'] == status].sample(
            n=n_to_take, random_state=42
        )
        sampled_parts.append(sampled)
        print(f"  {status:20s}: took {n_to_take:,} of {count:,}")

    # 3. Combine fraud + sampled non-fraud
    sampled_df = pd.concat([fraud_rows] + sampled_parts, ignore_index=True)

    # 4. Shuffle so fraud rows aren't all at the top
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Final sample size: {len(sampled_df):,} rows")
    print(f"  Fraud rows in sample: {sampled_df['fraud_label'].sum():,}")

    return sampled_df


# =============================================================================
# STEP 8 — LABEL ENCODING (text → numbers)
# =============================================================================
def label_encode(df):
    """
    ML models cannot work with text like 'Standard Class' or 'Consumer'.
    We convert each text category into a number using LabelEncoder.

    Example:
      'Africa'       → 0
      'Europe'       → 1
      'LATAM'        → 2
      'Pacific Asia' → 3
      'USCA'         → 4

    IMPORTANT: We save each encoder to disk (models/ folder).
    When M4 Agents make predictions on new data, they MUST use the
    SAME encoder — otherwise the numbers will mean different things!
    """
    print("\n" + "="*60)
    print("STEP 8: Label encoding categorical columns...")
    print("="*60)

    # These are the columns we encode (text → integer)
    # We create a NEW encoded column and keep the original for reference
    columns_to_encode = {
        'Type'              : 'type_enc',
        'Delivery Status'   : 'delivery_status_enc',
        'Customer Segment'  : 'customer_segment_enc',
        'Department Name'   : 'department_enc',
        'Market'            : 'market_enc',
        'Order Region'      : 'order_region_enc',
        'Order Status'      : 'order_status_enc',
        'Shipping Mode'     : 'shipping_mode_enc',
        'Category Name'     : 'category_enc',
        'Product Name'      : 'product_name_enc',
    }

    encoders = {}   # store all encoders in a dictionary

    for original_col, encoded_col in columns_to_encode.items():
        if original_col not in df.columns:
            print(f"  WARNING: Column '{original_col}' not found, skipping.")
            continue

        le = LabelEncoder()
        df[encoded_col] = le.fit_transform(df[original_col].astype(str))
        encoders[original_col] = le

        # Print the mapping so you can understand what numbers mean
        mapping = dict(zip(le.classes_, le.transform(le.classes_))) # type: ignore
        # mapping = dict(zip(le.classes_.tolist(), le.transform(le.classes_.tolist())))
        print(f"  '{original_col}' → '{encoded_col}' | {len(le.classes_)} classes")
        if len(le.classes_) <= 10:
            for text_val, num_val in sorted(mapping.items(), key=lambda x: x[1]):
                print(f"      {num_val}  =  {text_val}")

    # Save ALL encoders together in one file
    encoder_path = os.path.join(MODELS_DIR, "label_encoders.pkl")
    joblib.dump(encoders, encoder_path)
    print(f"\n  All encoders saved to: {encoder_path}")
    print(f"  To reload later: encoders = joblib.load('{encoder_path}')")

    return df, encoders


# =============================================================================
# STEP 9 — NORMALIZATION / SCALING (numbers → same scale)
# =============================================================================
def normalize_features(df):
    """
    Different numeric columns have very different scales:
      - Sales: ranges from 9 to 2000
      - Order Item Quantity: ranges from 1 to 5
      - Delay_days: ranges from -4 to 2

    If we feed these directly to ML models, Sales will dominate
    just because its numbers are bigger — even if it's not more important.

    We apply two types of scaling and save both:

    StandardScaler  → transforms to mean=0, std=1
                      Formula: (value - mean) / std
                      USE FOR: ML models like RandomForest, SVM, regression
                      (these don't require it but it helps convergence)

    MinMaxScaler    → transforms to range [0, 1]
                      Formula: (value - min) / (max - min)
                      USE FOR: Neural networks, visualization, Prophet inputs
    """
    print("\n" + "="*60)
    print("STEP 9: Normalizing numeric features...")
    print("="*60)

    # Columns to normalize
    # We do NOT normalize encoded columns (already integers)
    # We do NOT normalize binary columns (0/1 — already in range)
    scale_cols = [
        'Days for shipping (real)',
        'Days for shipment (scheduled)',
        'Order Item Quantity',
        'Order Item Discount Rate',
        'Order Item Profit Ratio',
        'Sales',
        'Order Profit Per Order',
        'Order Item Total',
        'Benefit per order',
        'Product Price',
        'Order Item Product Price',
        'Order Item Discount',
        'Sales per customer',
        'delay_days',
        'profit_margin',
        'discount_amount',
    ]

    # Keep only columns that actually exist in the dataframe
    scale_cols = [c for c in scale_cols if c in df.columns]

    # ── StandardScaler ───────────────────────────────────────────────────────
    std_scaler = StandardScaler()
    scaled_std = std_scaler.fit_transform(df[scale_cols])
    # Create new columns with _std suffix
    std_col_names = [c + '_std' for c in scale_cols]
    df_std = pd.DataFrame(scaled_std, columns=std_col_names, index=df.index)
    df = pd.concat([df, df_std], axis=1)

    # ── MinMaxScaler ─────────────────────────────────────────────────────────
    mm_scaler = MinMaxScaler()
    scaled_mm = mm_scaler.fit_transform(df[scale_cols])
    # Create new columns with _norm suffix
    norm_col_names = [c + '_norm' for c in scale_cols]
    df_norm = pd.DataFrame(scaled_mm, columns=norm_col_names, index=df.index)
    df = pd.concat([df, df_norm], axis=1)

    # Save both scalers
    std_path = os.path.join(MODELS_DIR, "standard_scaler.pkl")
    mm_path  = os.path.join(MODELS_DIR, "minmax_scaler.pkl")
    joblib.dump({'scaler': std_scaler, 'columns': scale_cols}, std_path)
    joblib.dump({'scaler': mm_scaler,  'columns': scale_cols}, mm_path)

    print(f"  Scaled {len(scale_cols)} columns")
    print(f"  Created: _std columns (StandardScaler) → use in M2 ML models")
    print(f"  Created: _norm columns (MinMaxScaler)  → use in Prophet / dashboard")
    print(f"  StandardScaler saved: {std_path}")
    print(f"  MinMaxScaler saved:   {mm_path}")

    return df, std_scaler, mm_scaler, scale_cols


# =============================================================================
# STEP 10 — BUILD WEEKLY AGGREGATION (for Demand Forecasting)
# =============================================================================
def build_weekly_demand(df):
    """
    The DemandAgent (M4) uses a Random Forest + Prophet model.
    These models need time-series data: quantity sold per product per week.

    This step aggregates the data:
      From: one row per order item
      To:   one row per (product, week, year) with total quantity

    Also computes lag features:
      lag_1 = quantity sold in the PREVIOUS week
      lag_2 = quantity sold 2 weeks ago
    These help the model predict based on recent trends.
    """
    print("\n" + "="*60)
    print("STEP 10: Building weekly demand aggregation for forecasting...")
    print("="*60)

    weekly = df.groupby(
        ['Product Name', 'product_name_enc', 'category_enc', 'market_enc',
         'order_year', 'order_week', 'order_month', 'order_season']
    ).agg(
        weekly_qty    = ('Order Item Quantity', 'sum'),
        weekly_sales  = ('Sales', 'sum'),
        weekly_orders = ('Order Id', 'count'),
    ).reset_index()

    # Sort so lag features work correctly (must be in time order)
    weekly = weekly.sort_values(['Product Name', 'order_year', 'order_week'])

    # Lag features — shift by 1 and 2 rows within each product group
    weekly['lag_1'] = weekly.groupby('Product Name')['weekly_qty'].shift(1)
    weekly['lag_2'] = weekly.groupby('Product Name')['weekly_qty'].shift(2)

    # Fill the first 1-2 rows of each product (which have no lag) with 0
    weekly['lag_1'] = weekly['lag_1'].fillna(0)
    weekly['lag_2'] = weekly['lag_2'].fillna(0)

    # Also add a ds column (date string for Prophet: format YYYY-MM-DD)
    # We approximate date from year + week
    weekly['ds'] = pd.to_datetime(
        weekly['order_year'].astype(str) + '-W' +
        weekly['order_week'].astype(str).str.zfill(2) + '-1',
        format='%Y-W%W-%w',
        errors='coerce'
    )

    weekly_path = os.path.join(PROC_DIR, "weekly_demand.csv")
    weekly.to_csv(weekly_path, index=False)

    print(f"  Weekly demand rows: {len(weekly):,}")
    print(f"  Products covered: {weekly['Product Name'].nunique():,}")
    print(f"  Saved: {weekly_path}")

    return weekly


# =============================================================================
# STEP 11 — PRECOMPUTE ACCESS LOG BROWSE SIGNALS
# =============================================================================
def precompute_browse_signals():
    """
    The tokenized_access_logs.csv tells us which products/categories
    were browsed the most. We use this as a demand signal in XAI:

    Example:  "Demand alert for Cleats — ALSO: Cleats is the #1
               browsed category in access logs (27,878 views)"

    We just count browse frequency per category and save the top rankings.
    This is a lightweight one-time computation — no need to load 469K rows
    every time the agents run.
    """
    print("\n" + "="*60)
    print("STEP 11: Precomputing browse signals from access logs...")
    print("="*60)

    try:
        logs = pd.read_csv(LOGS_DATA, encoding='latin-1')

        # Clean department/category columns (strip whitespace)
        logs['Department'] = logs['Department'].str.strip().str.lower()
        logs['Category']   = logs['Category'].str.strip().str.lower()

        # Count browsing frequency
        cat_browse  = logs['Category'].value_counts().reset_index()
        dept_browse = logs['Department'].value_counts().reset_index()
        cat_browse.columns  = ['category', 'browse_count']
        dept_browse.columns = ['department', 'browse_count']

        # Add rank column
        cat_browse['browse_rank']  = range(1, len(cat_browse) + 1)
        dept_browse['browse_rank'] = range(1, len(dept_browse) + 1)

        # Save
        cat_path  = os.path.join(PROC_DIR, "browse_signals_category.csv")
        dept_path = os.path.join(PROC_DIR, "browse_signals_department.csv")
        cat_browse.to_csv(cat_path,   index=False)
        dept_browse.to_csv(dept_path, index=False)

        print(f"  Top 5 browsed categories:")
        for _, row in cat_browse.head(5).iterrows():
            print(f"    #{row['browse_rank']:2d}  {row['category']:30s}  {row['browse_count']:,} views")

        print(f"  Saved: {cat_path}")
        print(f"  Saved: {dept_path}")

        return cat_browse, dept_browse

    except FileNotFoundError:
        print(f"  WARNING: Access logs not found at {LOGS_DATA}. Skipping.")
        return None, None


# =============================================================================
# STEP 12 — SAVE FINAL CLEAN DATASET
# =============================================================================
def save_clean_data(df):
    """
    Save the final processed dataset.
    This is the file all other modules will load.
    """
    print("\n" + "="*60)
    print("STEP 12: Saving clean dataset...")
    print("="*60)

    output_path = os.path.join(PROC_DIR, "clean_data.csv")
    df.to_csv(output_path, index=False)

    print(f"  Saved: {output_path}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n  Columns in final dataset:")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:3d}. {col}")


# =============================================================================
# STEP 13 — PRINT FINAL SUMMARY
# =============================================================================
def print_summary(df, inventory):
    """
    Print a final summary of everything that was done.
    This is your sanity check — read it and confirm everything looks right.
    """
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE — FINAL SUMMARY")
    print("="*60)
    print(f"\n  Clean dataset rows   : {len(df):,}")
    print(f"  Clean dataset columns: {df.shape[1]}")
    print(f"  Fraud rows           : {df['fraud_label'].sum():,}")
    print(f"  Non-fraud rows       : {(df['fraud_label']==0).sum():,}")
    print(f"  Date range           : {df['order date (DateOrders)'].min().date()} "
          f"to {df['order date (DateOrders)'].max().date()}")
    print(f"\n  Inventory products   : {len(inventory):,}")
    print(f"  CRITICAL stock items : {(inventory['status']=='CRITICAL').sum()}")
    print(f"  WARNING stock items  : {(inventory['status']=='WARNING').sum()}")
    print(f"\n  Files saved to data/processed/:")
    for f in os.listdir(PROC_DIR):
        size_kb = os.path.getsize(os.path.join(PROC_DIR, f)) // 1024
        print(f"    {f:40s} {size_kb:6} KB")
    print(f"\n  Encoder/scaler files saved to models/:")
    for f in os.listdir(MODELS_DIR):
        size_kb = os.path.getsize(os.path.join(MODELS_DIR, f)) // 1024
        print(f"    {f:40s} {size_kb:6} KB")
    print()


# =============================================================================
# MAIN — Run all steps in order
# =============================================================================
def run_preprocessing():
    """
    This is the main function.
    Call this to run the entire preprocessing pipeline from start to finish.

    NOTE: stratified_sample() has been intentionally removed.
    The full 180K-row dataset is used for all steps because:
      - Preprocessing takes ~6 seconds on the full dataset
      - ML models train once and save to .pkl — runtime cost is one-time
      - Sampling to ~5K rows was creating an 80% fraud / 20% legit imbalance
        (vs the real 2.3% fraud rate), which corrupted the fraud model
      - Weekly demand built from 5K rows gave only ~18 weeks of history per
        product; the full dataset gives ~60 weeks — 3x richer for Prophet
    """
    print("\n" + "#"*60)
    print("  AutoBiz AI — MODULE 1: DATA PREPROCESSING")
    print("#"*60)

    # Run each step in order
    df           = load_raw_data()
    df           = drop_columns(df)
    df           = fix_datatypes(df)
    df           = handle_missing_values(df)
    df           = feature_engineering(df)
    inventory    = build_inventory_table(df)
    # NOTE: stratified_sample() removed — full dataset used (see docstring above)
    df, encoders = label_encode(df)
    df, std_sc, mm_sc, scale_cols = normalize_features(df)
    weekly       = build_weekly_demand(df)
    precompute_browse_signals()
    save_clean_data(df)
    print_summary(df, inventory)

    # Return everything so other modules can import and reuse
    return df, inventory, weekly, encoders, std_sc, mm_sc


# =============================================================================
# ENTRY POINT
# If you run this file directly (python modules/preprocessing.py),
# it will execute run_preprocessing().
# =============================================================================
if __name__ == "__main__":
    
    df, inventory, weekly, encoders, std_scaler, mm_scaler = run_preprocessing()

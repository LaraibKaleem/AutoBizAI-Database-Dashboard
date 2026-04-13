
# AutoBiz AI 🤖 — Autonomous Business Decision Agents
> Autonomous Business Decision Agents — AI-powered supply chain analytics
> with demand forecasting, fraud detection, and inventory risk management.

### 📑 Module Description

| Module | Description |
|---|---|
| M1 — Data Preprocessing | Cleans raw data, feature engineering |
| M2 — Machine Learning | Demand, fraud, inventory models |
| M3 — Database & Backend | Turso cloud DB, all data management |
| M4 — Agents + Dashboard | AI agents, XAI, Streamlit dashboard |


### ⚙️ Tech Stack
- **Python** 3.14 
- **Turso** — cloud SQLite database (Turso.tech)
- **Prophet** — demand forecasting
- **XGBoost + RandomForest** — fraud & inventory models
- **Streamlit** — interactive dashboard
- **httpx** — Turso HTTP API calls
- **scikit-learn** — ML preprocessing & evaluation

### 📦 Files Summary
| Module | File | Purpose |
|--------|------|---------|
| M1 | modules/preprocessing.py | Load + clean DataCo dataset |
| M2 | modules/machinelearning.py | Train 3 ML models |
| M3 | modules/database.py | All Turso DB operations |
| M4 | agents/*.py | AI decision agents |
| M5 | app.py | Streamlit dashboard |
| M6 | tests/test_integration.py | Integration tests |

### 🗄️ Database Description

##### 🗄️ DataSet - DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS
https://data.mendeley.com/datasets/8gx2fvg2k6/3


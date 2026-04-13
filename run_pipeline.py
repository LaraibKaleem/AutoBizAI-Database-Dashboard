# =============================================================================
# run_pipeline.py
# Run everything in one go:
#   python run_pipeline.py
# =============================================================================

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable

def run(label, script):
    print(f"\n{'='*55}")
    print(f"  ▶  {label}")
    print(f"{'='*55}")
    result = subprocess.run(
        [PYTHON, os.path.join(BASE_DIR, script)],
        cwd=BASE_DIR
    )
    if result.returncode != 0:
        print(f"\n  ❌ FAILED: {label}")
        print(f"  Fix the error above then re-run.")
        sys.exit(1)
    print(f"  ✅ Done: {label}")


if __name__ == "__main__":
    print("\n" + "#"*55)
    print("  AutoBiz AI — Full Pipeline Runner")
    print("#"*55)

    # Step 1 — Preprocessing
    run("Module 1 — Data Preprocessing",   "modules/preprocessing.py")

    # Step 2 — Database setup
    run("Module 3 — Database Init",         "init_db.py")

    # Step 3 — ML Models
    run("Module 2 — Machine Learning",      "modules/machinelearning.py")

    # Step 4 — AI Agents
    run("Module 4 — Run All Agents",        "agents/run_all_agents.py")

    # Step 5 — Integration Tests
    run("Module 6 — Integration Tests",     "tests/test_integration.py")

    print("\n" + "#"*55)
    print("  ✅ ALL STEPS COMPLETE")
    print("  Next: streamlit run app.py")
    print("#"*55)
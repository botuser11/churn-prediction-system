import json
from pathlib import Path

def read_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)

def load_app_metadata():
    meta = {}

    # Phase 2 training summary
    train_summary = read_json("reports/train_final_phase2.json")
    if train_summary:
        meta["train_summary"] = train_summary

    # Phase 2 threshold tuning + test metrics
    phase2_final = read_json("reports/final_metrics_phase2.json")
    if phase2_final:
        meta["phase2_test_metrics"] = phase2_final

    # Phase 3 cost threshold
    cost = read_json("reports/threshold_by_cost.json")
    if cost:
        meta["cost_threshold"] = cost

    return meta

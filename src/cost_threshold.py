import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

def expected_cost(cm, cost_fp: float, cost_fn: float):
    tn, fp, fn, tp = cm.ravel()
    return fp * cost_fp + fn * cost_fn

if __name__ == "__main__":
    data = np.load("reports/probas_lr.npz", allow_pickle=True)
    p_val = data["p_val"]
    y_val = data["y_val"]

    # You can adjust these later (business assumption)
    COST_FP = 10.0   # e.g., £10 retention offer wasted
    COST_FN = 100.0  # e.g., £100 lost LTV by missing a churner

    thresholds = np.linspace(0.05, 0.95, 181)
    rows = []
    best = None

    for t in thresholds:
        yhat = (p_val >= t).astype(int)
        cm = confusion_matrix(y_val, yhat, labels=[0,1])
        cost = expected_cost(cm, COST_FP, COST_FN)

        rows.append({
            "threshold": float(t),
            "tn": int(cm[0,0]),
            "fp": int(cm[0,1]),
            "fn": int(cm[1,0]),
            "tp": int(cm[1,1]),
            "expected_cost": float(cost),
        })

        if best is None or cost < best["expected_cost"]:
            best = rows[-1]

    out = {
        "tuning_on": "val",
        "cost_fp": COST_FP,
        "cost_fn": COST_FN,
        "best_threshold_by_cost": best,
        "cost_curve": rows,
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/threshold_by_cost.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({
        "best_threshold_by_cost": best,
        "cost_fp": COST_FP,
        "cost_fn": COST_FN,
    }, indent=2))

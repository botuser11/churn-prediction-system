import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

if __name__ == "__main__":
    data = np.load("reports/probas_lr.npz", allow_pickle=True)
    p_val = data["p_val"]
    y_val = data["y_val"]
    p_test = data["p_test"]
    y_test = data["y_test"]

    thresholds = np.linspace(0.2, 0.8, 25)
    rows = []
    for t in thresholds:
        yhat = (p_val >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_val, yhat, zero_division=0)),
            "recall": float(recall_score(y_val, yhat, zero_division=0)),
            "f1": float(f1_score(y_val, yhat, zero_division=0)),
        })

    df_thr = pd.DataFrame(rows)
    best = df_thr.iloc[df_thr["f1"].argmax()].to_dict()
    t_best = float(best["threshold"])

    # Evaluate on TEST using the threshold chosen on VAL
    yhat_test = (p_test >= t_best).astype(int)
    cm_test = confusion_matrix(y_test, yhat_test).tolist()

    test_metrics = {
        "threshold": t_best,
        "precision": float(precision_score(y_test, yhat_test, zero_division=0)),
        "recall": float(recall_score(y_test, yhat_test, zero_division=0)),
        "f1": float(f1_score(y_test, yhat_test, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "confusion_matrix": cm_test,
    }

    out = {
        "tuning_on": "val",
        "threshold_table": rows,
        "best_by_f1_on_val": best,
        "test_metrics_at_best_threshold": test_metrics,
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/threshold_tuning_lr_phase2.json", "w") as f:
        json.dump(out, f, indent=2)

    # Also write a Phase-2 final metrics file (what you’ll cite in README/interviews)
    with open("reports/final_metrics_phase2.json", "w") as f:
        json.dump(out["test_metrics_at_best_threshold"], f, indent=2)

    print(json.dumps({
        "best_threshold_on_val": t_best,
        "test_metrics_at_best_threshold": test_metrics,
    }, indent=2))

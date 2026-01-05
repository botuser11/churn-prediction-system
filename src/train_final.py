import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.metrics import roc_auc_score
from pipeline_factory import make_lr_pipeline
from split_utils import make_splits, save_splits, apply_split

SEED = 42

def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

def load_data(cfg):
    df = pd.read_csv(cfg["data_path"])

    # Normalize blanks -> NaN (helps general datasets)
    df = df.replace(r"^\s*$", np.nan, regex=True)

    pos = str(cfg["positive_label"]).strip().lower()
    y = (
        df[cfg["target"]]
        .astype(str).str.strip().str.lower()
        .map({pos: 1})
        .fillna(0).astype(int)
    )

    X = df.drop(columns=[cfg["target"]] + cfg.get("drop_cols", []), errors="ignore")

    # Coerce numeric-like objects if possible (keeps non-numeric as object)
    for c in X.columns:
        if X[c].dtype == object:
            try:
                X[c] = pd.to_numeric(X[c])
            except Exception:
                pass


    return X, y

if __name__ == "__main__":
    cfg = load_config()
    X, y = load_data(cfg)

    # 1) Create and SAVE splits (one-time, deterministic)
    splits = make_splits(X, y, test_size=0.2, val_size=0.25, seed=SEED)
    # val_size=0.25 of trainval => overall ~20% val, 20% test, 60% train
    save_splits(splits)

    X_tr, y_tr, X_va, y_va, X_te, y_te = apply_split(X, y, splits)

    # 2) Train on TRAIN only
    pipe = make_lr_pipeline(X_tr, seed=SEED)
    pipe.fit(X_tr, y_tr)

    # 3) Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, cfg["model_path"])

    # 4) Save probabilities for val/test (for Phase 2 thresholding)
    Path("reports").mkdir(parents=True, exist_ok=True)
    p_val = pipe.predict_proba(X_va)[:, 1]
    p_test = pipe.predict_proba(X_te)[:, 1]

    np.savez(
        "reports/probas_lr.npz",
        p_val=p_val,
        y_val=y_va.to_numpy(),
        p_test=p_test,
        y_test=y_te.to_numpy()
    )
    ap_val = average_precision_score(y_va, p_val)
    ap_test = average_precision_score(y_te, p_test)

    brier_val = brier_score_loss(y_va, p_val)
    brier_test = brier_score_loss(y_te, p_test)

    # 5) Report threshold-independent metric (AUC) on val/test
    out = {
        "roc_auc_val": float(roc_auc_score(y_va, p_val)),
        "roc_auc_test": float(roc_auc_score(y_te, p_test)),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "model_path": cfg["model_path"],
        "split_path": "reports/split_indices.json",
        "probas_path": "reports/probas_lr.npz",
        "pr_auc_val": float(ap_val),
        "pr_auc_test": float(ap_test),
        "brier_val": float(brier_val),
        "brier_test": float(brier_test),
    }

    with open("reports/train_final_phase2.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Saved model to:", cfg["model_path"])
    print(json.dumps(out, indent=2))

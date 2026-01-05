import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED_DEFAULT = 42

def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,  # fraction of (train+val) after test is removed
    seed: int = SEED_DEFAULT,
):
    # Ensure we have stable integer indices
    idx = np.arange(len(X))

    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed
    )

    y_trainval = y.iloc[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_size, stratify=y_trainval, random_state=seed
    )

    splits = {
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "train_idx": idx_train.tolist(),
        "val_idx": idx_val.tolist(),
        "test_idx": idx_test.tolist(),
    }
    return splits

def save_splits(splits: dict, path: str = "reports/split_indices.json"):
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)

def load_splits(path: str = "reports/split_indices.json"):
    with open(path, "r") as f:
        return json.load(f)

def apply_split(X: pd.DataFrame, y: pd.Series, splits: dict):
    tr = splits["train_idx"]
    va = splits["val_idx"]
    te = splits["test_idx"]
    return (
        X.iloc[tr], y.iloc[tr],
        X.iloc[va], y.iloc[va],
        X.iloc[te], y.iloc[te],
    )

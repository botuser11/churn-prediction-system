import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from preprocess import load_and_clean

SEED = 42

def build_preprocessor(X: pd.DataFrame):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ]
    )

if __name__ == "__main__":
    df, _ = load_and_clean("data/raw/telco_churn.csv")
    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    pre = build_preprocessor(X_train)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=SEED)

    pipe = Pipeline([("pre", pre), ("clf", gb)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/gb_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Gradient Boosting Results:")
    print(json.dumps(metrics, indent=2))

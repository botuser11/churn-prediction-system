import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression

from preprocess import load_and_clean
from pipeline_factory import make_lr_pipeline

SEED = 42

def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

if __name__ == "__main__":
    # 1) Load & split
    df, _ = load_and_clean("data/raw/telco_churn.csv")
    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # 2) Preprocessor + model
    preprocessor = build_preprocessor(X_train)
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)

    pipe = make_lr_pipeline(X_train, seed=SEED)
    pipe.fit(X_train, y_train)


    # 3) Fit
    pipe.fit(X_train, y_train)

    # 4) Predict + metrics
    y_pred = pipe.predict(X_test)
    # Probabilities for ROC-AUC
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        # fallback if predict_proba not available
        y_proba = y_pred.astype(float)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }
    cm = confusion_matrix(y_test, y_pred).tolist()

    Path("reports").mkdir(parents=True, exist_ok=True)
    out = {"metrics": metrics, "confusion_matrix": cm}
    with open("reports/baseline_lr_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Baseline Logistic Regression")
    print(json.dumps(out, indent=2))

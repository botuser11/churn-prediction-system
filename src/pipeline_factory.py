import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def split_num_cat(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = split_num_cat(X)

    numeric_transformer = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor

def make_lr_pipeline(X_train: pd.DataFrame, seed: int = 42) -> Pipeline:
    pre = build_preprocessor(X_train)
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])
    return pipe

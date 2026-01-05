import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from preprocess import load_and_clean

def build_preprocessor(df: pd.DataFrame, target: str = "Churn"):
    # Split features into numeric and categorical
    y = df[target]
    X = df.drop(columns=[target])

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ColumnTransformer to apply each set
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return X, y, preprocessor

if __name__ == "__main__":
    df, missing = load_and_clean("data/raw/telco_churn.csv")
    X, y, preprocessor = build_preprocessor(df)

    print("Numeric features:", len(X.select_dtypes(include=[np.number]).columns))
    print("Categorical features:", len(X.select_dtypes(exclude=[np.number]).columns))
    print("Example categorical cols:", X.select_dtypes(exclude=[np.number]).columns[:5].tolist())

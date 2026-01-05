import pandas as pd
import numpy as np

def load_and_clean(path: str):
    df = pd.read_csv(path)

    # 1. Standardize target
    df["Churn"] = df["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

    # 2. Coerce TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3. Drop customerID (identifier, no predictive value)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 4. Report missing counts
    missing_counts = df.isna().sum()

    return df, missing_counts

if __name__ == "__main__":
    df, missing = load_and_clean("data/raw/telco_churn.csv")
    print("Shape after cleaning:", df.shape)
    print("\nTarget distribution:")
    print(df["Churn"].value_counts())
    print("\nMissing values per column:")
    print(missing[missing > 0])

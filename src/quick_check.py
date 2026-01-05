import pandas as pd
from pathlib import Path

csv_path = Path("data/raw/telco_churn.csv")
df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nTarget preview (unique values in 'Churn'):")
print(df["Churn"].astype(str).str.strip().value_counts(dropna=False).head(10))

print("\nDtypes summary:")
print(df.dtypes.head(15))

# Common gotcha: TotalCharges sometimes comes as string—check quickly
if "TotalCharges" in df.columns:
    bad = pd.to_numeric(df["TotalCharges"], errors="coerce").isna().sum()
    print(f"\nTotalCharges non-numeric count (after coercion attempt): {bad}")

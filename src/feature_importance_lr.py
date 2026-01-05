import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = "models/final_model.joblib"

if __name__ == "__main__":
    pipe = joblib.load(MODEL_PATH)
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    feature_names = pre.get_feature_names_out()
    coefs = clf.coef_.ravel()

    df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    Path("reports").mkdir(parents=True, exist_ok=True)
    df.to_csv("reports/feature_importance_lr.csv", index=False)

    # also save top 20 in each direction for quick inspection
    top_pos = df.sort_values("coef", ascending=False).head(20)
    top_neg = df.sort_values("coef", ascending=True).head(20)

    top_pos.to_csv("reports/top_positive_drivers.csv", index=False)
    top_neg.to_csv("reports/top_negative_drivers.csv", index=False)

    print("Saved:")
    print("- reports/feature_importance_lr.csv")
    print("- reports/top_positive_drivers.csv")
    print("- reports/top_negative_drivers.csv")

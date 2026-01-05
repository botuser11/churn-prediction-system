import joblib
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from preprocess import load_and_clean
from train_gb import build_preprocessor, SEED
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    # 1) Load data
    df, _ = load_and_clean("data/raw/telco_churn.csv")
    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # 2) Rebuild preprocessor + model (must match train_gb.py)
    pre = build_preprocessor(X_train)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=SEED)
    pipe = Pipeline([("pre", pre), ("clf", gb)])
    pipe.fit(X_train, y_train)

    # 3) Get feature names from preprocessor
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
    cat_features = ohe.get_feature_names_out(cat_cols)

    feature_names = np.concatenate([num_cols, cat_features])
    importances = pipe.named_steps["clf"].feature_importances_

    # 4) Sort by importance
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(15)

    # 5) Save + plot
    fi.to_csv("reports/feature_importance.csv", index=False)

    plt.figure(figsize=(8,6))
    plt.barh(fi["feature"], fi["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 15 Features Driving Churn")
    plt.tight_layout()
    plt.savefig("reports/figs/feature_importance.png", dpi=150)
    plt.close()

    print("Saved top features to reports/feature_importance.csv and reports/figs/feature_importance.png")
    print(fi)

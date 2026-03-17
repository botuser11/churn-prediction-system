import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
 
 
def get_feature_names(pipe: Pipeline) -> list[str]:
    """Extract feature names after preprocessing (post one-hot encoding)."""
    pre = pipe.named_steps["pre"]
    return list(pre.get_feature_names_out())
 
 
def build_shap_explainer(pipe: Pipeline, X_background: pd.DataFrame):
    """
    Build a SHAP LinearExplainer for a logistic regression pipeline.
    X_background: a sample of raw training data used to estimate feature
    distributions. 100-200 rows is sufficient.
    """
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    X_bg_transformed = pre.transform(X_background)
    explainer = shap.LinearExplainer(
        clf,
        X_bg_transformed,
        feature_perturbation="correlation_dependent",
    )
    return explainer
 
 
def explain_single(
    pipe: Pipeline,
    explainer: shap.LinearExplainer,
    df_single: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return a DataFrame of the top_n SHAP values for a single customer record.
    Columns: feature_clean, shap_value, abs_shap, direction
    Sorted by absolute SHAP value descending.
    """
    pre = pipe.named_steps["pre"]
    feature_names = list(pre.get_feature_names_out())
 
    X_transformed = pre.transform(df_single)
    shap_values = explainer.shap_values(X_transformed)
 
    vals = shap_values[0] if shap_values.ndim == 2 else shap_values
 
    df_shap = pd.DataFrame({
        "feature": feature_names,
        "shap_value": vals,
        "abs_shap": np.abs(vals),
    }).sort_values("abs_shap", ascending=False).head(top_n)
 
    df_shap["direction"] = df_shap["shap_value"].apply(
        lambda x: "increases churn risk" if x > 0 else "reduces churn risk"
    )
 
    # Clean sklearn prefixes for display
    df_shap["feature_clean"] = (
        df_shap["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
 
    return df_shap.reset_index(drop=True)
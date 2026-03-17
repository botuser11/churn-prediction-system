import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
 
from src.model_metadata import load_app_metadata
from src.shap_explainer import build_shap_explainer, explain_single
 
st.set_page_config(page_title="Churn Predictor", page_icon="📉")
 
# ----------------------------
# Config
# ----------------------------
CFG_PATH = Path("config.json")
with open(CFG_PATH, "r") as f:
    CFG = json.load(f)
 
if "ACTIVE_MODEL_PATH" not in st.session_state:
    st.session_state.ACTIVE_MODEL_PATH = CFG["model_path"]
 
META = load_app_metadata()
 
 
# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)
 
@st.cache_data
def load_drivers():
    pos_path = Path("reports/top_positive_drivers.csv")
    neg_path = Path("reports/top_negative_drivers.csv")
    pos = pd.read_csv(pos_path) if pos_path.exists() else None
    neg = pd.read_csv(neg_path) if neg_path.exists() else None
    return pos, neg
 
@st.cache_resource
def load_shap_explainer(model_path: str):
    """Build SHAP explainer using a background sample from the training data."""
    pipe = load_model(model_path)
    df_raw = pd.read_csv(CFG["data_path"])
    df_raw = df_raw.replace(r"^\s*$", np.nan, regex=True)
 
    drop = [CFG["target"]] + CFG.get("drop_cols", [])
    X_all = df_raw.drop(columns=drop, errors="ignore")
 
    for c in X_all.columns:
        if X_all[c].dtype == object:
            try:
                X_all[c] = pd.to_numeric(X_all[c])
            except Exception:
                pass
 
    # Use up to 200 rows as background sample
    bg = X_all.sample(min(200, len(X_all)), random_state=42)
    explainer = build_shap_explainer(pipe, bg)
    return explainer
 
 
# ----------------------------
# Helpers: schema + input alignment
# ----------------------------
def get_expected_schema(pipe):
    pre = pipe.named_steps["pre"]
    name_to_cols = {name: cols for (name, _, cols) in pre.transformers_}
    num_cols = list(name_to_cols.get("num", []))
    cat_cols = list(name_to_cols.get("cat", []))
    return num_cols, cat_cols
 
def align_columns(df, expected_cols):
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[expected_cols]
 
def schema_warnings(input_df: pd.DataFrame, expected_cols: list[str]):
    missing = [c for c in expected_cols if c not in input_df.columns]
    extra = [c for c in input_df.columns if c not in expected_cols]
    missing_rate = len(missing) / max(1, len(expected_cols))
    if missing_rate > 0.30:
        st.warning(
            f"Input is missing {len(missing)}/{len(expected_cols)} expected columns "
            f"({missing_rate:.0%}). Predictions may be unreliable."
        )
        with st.expander("Show missing columns"):
            st.write(missing[:200])
    if len(extra) > 0:
        st.caption(f"Ignoring {len(extra)} extra columns not used by the model.")
 
 
def predict_df(df: pd.DataFrame, pipe: Pipeline, threshold: float):
    num_cols, cat_cols = get_expected_schema(pipe)
    expected = num_cols + cat_cols
 
    df_clean = df.copy()
    df_clean = df_clean.replace(r"^\s*$", np.nan, regex=True)
 
    for c in num_cols:
        if c in df_clean.columns:
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
 
    schema_warnings(df_clean, expected)
    df_for_model = align_columns(df_clean, expected)
 
    proba = pipe.predict_proba(df_for_model)[:, 1]
    pred = (proba >= threshold).astype(int)
 
    out = df.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = pred
    return out
 
 
# ----------------------------
# Sidebar
# ----------------------------
PIPE = load_model(st.session_state.ACTIVE_MODEL_PATH)
 
with st.sidebar:
    st.markdown(f"**Active model:** `{st.session_state.ACTIVE_MODEL_PATH}`")
 
    st.subheader("Decision Threshold")
 
    modes = ["Manual", "F1 (Validation-tuned)", "Cost (Min expected cost)"]
    default_mode = 0
    st.session_state.THRESH_MODE = st.selectbox("Threshold mode", modes, index=default_mode)
 
    f1_thr = None
    if META.get("phase2_test_metrics") and "threshold" in META["phase2_test_metrics"]:
        f1_thr = float(META["phase2_test_metrics"]["threshold"])
 
    cost_thr = None
    if META.get("cost_threshold") and "best_threshold_by_cost" in META["cost_threshold"]:
        cost_thr = float(META["cost_threshold"]["best_threshold_by_cost"]["threshold"])
 
    if "MANUAL_THRESHOLD" not in st.session_state:
        st.session_state.MANUAL_THRESHOLD = float(CFG.get("threshold", 0.5))
    if "ACTIVE_THRESHOLD" not in st.session_state:
        st.session_state.ACTIVE_THRESHOLD = float(st.session_state.MANUAL_THRESHOLD)
 
    if st.session_state.THRESH_MODE == "Manual":
        st.session_state.MANUAL_THRESHOLD = st.slider(
            "Flag as positive when probability ≥",
            min_value=0.05, max_value=0.95,
            value=float(st.session_state.MANUAL_THRESHOLD),
            step=0.01,
        )
        st.session_state.ACTIVE_THRESHOLD = float(st.session_state.MANUAL_THRESHOLD)
    elif st.session_state.THRESH_MODE == "F1 (Validation-tuned)":
        if f1_thr is None:
            st.warning("No Phase 2 tuned threshold found.")
            st.session_state.ACTIVE_THRESHOLD = float(st.session_state.MANUAL_THRESHOLD)
        else:
            st.session_state.ACTIVE_THRESHOLD = float(f1_thr)
            st.info(f"Using F1-tuned threshold: {float(f1_thr):.3f}")
    else:
        if cost_thr is None:
            st.warning("No cost-threshold found.")
            st.session_state.ACTIVE_THRESHOLD = float(st.session_state.MANUAL_THRESHOLD)
        else:
            st.session_state.ACTIVE_THRESHOLD = float(cost_thr)
            st.info(f"Using cost-optimal threshold: {float(cost_thr):.3f}")
 
    THRESHOLD = float(st.session_state.ACTIVE_THRESHOLD)
 
    st.divider()
    st.subheader("Prediction Context")
    st.write(f"**Model:** `{Path(st.session_state.ACTIVE_MODEL_PATH).name}`")
    st.write(f"**Threshold mode:** {st.session_state.THRESH_MODE}")
    st.write(f"**Active threshold:** {THRESHOLD:.3f}")
 
    st.divider()
    st.subheader("Model Metrics (Test)")
    train_sum = META.get("train_summary")
    if train_sum:
        st.write(f"Train/Val/Test: {train_sum['n_train']}/{train_sum['n_val']}/{train_sum['n_test']}")
        if train_sum.get("roc_auc_test") is not None:
            st.write(f"ROC-AUC (test): {float(train_sum['roc_auc_test']):.3f}")
        if train_sum.get("pr_auc_test") is not None:
            st.write(f"PR-AUC  (test): {float(train_sum['pr_auc_test']):.3f}")
        if train_sum.get("brier_test") is not None:
            st.write(f"Brier   (test): {float(train_sum['brier_test']):.3f}")
    else:
        st.caption("Run Phase 2/3 training to populate metrics.")
 
    st.divider()
    st.subheader("Top Drivers")
    pos, neg = load_drivers()
    if pos is not None and neg is not None:
        st.caption("Positive = increases churn risk. Negative = reduces churn risk.")
        st.write("Top + drivers")
        st.dataframe(pos.head(10), use_container_width=True)
        st.write("Top - drivers")
        st.dataframe(neg.head(10), use_container_width=True)
    else:
        st.caption("Run Phase 3 feature importance export to populate reports/top_*_drivers.csv")
 
 
# ----------------------------
# Main UI
# ----------------------------
st.title("📉 Customer Churn Predictor")
st.caption("Logistic Regression pipeline with automatic preprocessing")
 
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Prediction",
    "Batch CSV Prediction",
    "Train New Model",
    "Explain Prediction"
])
 
 
# ----------------------------
# Tab 1: Single prediction
# ----------------------------
with tab1:
    st.subheader("Single customer (paste one JSON record)")
    example = (
        '{\n'
        '  "gender": "Female", "SeniorCitizen": 0,\n'
        '  "Partner": "Yes", "Dependents": "No",\n'
        '  "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",\n'
        '  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",\n'
        '  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",\n'
        '  "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",\n'
        '  "MonthlyCharges": 80.5, "TotalCharges": 970.2\n'
        '}'
    )
    json_input = st.text_area("JSON input", value=example, height=260)
 
    if st.button("Predict (single)"):
        try:
            record = json.loads(json_input)
            df = pd.DataFrame([record])
            res = predict_df(df, PIPE, THRESHOLD)
            st.success(
                f"Churn probability: {res.loc[0, 'churn_proba']:.3f}  •  Pred: {int(res.loc[0, 'churn_pred'])}"
            )
            st.dataframe(res, use_container_width=True)
        except Exception as e:
            st.error(f"Failed: {e}")
 
 
# ----------------------------
# Tab 2: Batch CSV
# ----------------------------
with tab2:
    st.subheader("Batch CSV prediction")
    up = st.file_uploader(
        "Upload CSV with the same feature columns used in training (no target column)",
        type=["csv"]
    )
 
    if up is not None:
        try:
            df = pd.read_csv(up, sep=None, engine="python")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()
 
        st.write("Preview:", df.head())
 
        if st.button("Predict (batch)"):
            try:
                res = predict_df(df, PIPE, THRESHOLD)
                st.success("Predicted!")
 
                available_cols = res.columns.tolist()
                id_col = st.selectbox(
                    "Optional ID column (for display/sorting)",
                    ["<none>"] + available_cols, index=0,
                )
                top_k = st.slider("Show top N highest-risk rows", min_value=5, max_value=200, value=20)
                simple_view = st.checkbox("Simple view (probability only)", value=True)
 
                n_rows = len(res)
                pos_rate = float((res["churn_pred"] == 1).mean())
                st.write(
                    f"**Rows:** {n_rows:,}  •  **Threshold:** {THRESHOLD:.3f}  •  "
                    f"**% flagged positive:** {pos_rate:.2%}"
                )
 
                fig, ax = plt.subplots()
                ax.hist(res["churn_proba"], bins=30)
                ax.axvline(THRESHOLD, linestyle="--")
                ax.set_xlabel("Predicted probability")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of predicted probabilities")
                st.pyplot(fig)
 
                show_cols = []
                if id_col != "<none>":
                    show_cols.append(id_col)
                show_cols += ["churn_proba", "churn_pred"]
 
                top_view = (
                    res.sort_values("churn_proba", ascending=False)
                    .loc[:, show_cols]
                    .head(top_k)
                )
                st.subheader("Top at-risk customers")
                st.dataframe(top_view, use_container_width=True)
 
                if not simple_view:
                    context_cols = st.multiselect(
                        "Add context columns to the table (optional)",
                        [c for c in available_cols if c not in show_cols],
                        default=[]
                    )
                    if id_col != "<none>":
                        adv_cols = show_cols[:-2] + context_cols + ["churn_proba", "churn_pred"]
                    else:
                        adv_cols = context_cols + show_cols
 
                    st.dataframe(
                        res.sort_values("churn_proba", ascending=False)
                        .loc[:, adv_cols].head(200),
                        use_container_width=True
                    )
 
                safe_mode = st.session_state.THRESH_MODE.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                fname = f"predictions_{Path(st.session_state.ACTIVE_MODEL_PATH).stem}_{safe_mode}_thr{THRESHOLD:.3f}.csv"
                st.download_button(f"Download {fname}", res.to_csv(index=False), fname, "text/csv")
 
            except Exception as e:
                st.error(f"Failed: {e}")
 
 
# ----------------------------
# Tab 3: Train new model
# ----------------------------
with tab3:
    st.subheader("Train a new model on an uploaded CSV (binary target)")
    up = st.file_uploader("Upload CSV", type=["csv"], key="traincsv")
    if up is not None:
        df = pd.read_csv(up, sep=None, engine="python")
        st.write("Preview:", df.head())
 
        target_col = st.selectbox("Target column", df.columns)
        pos_label = st.text_input("Positive label (e.g., Yes or 1)", value="Yes")
        drop_cols = st.multiselect("Columns to drop (IDs etc.)", df.columns.tolist())
 
        if st.button("Train Model"):
            try:
                df = df.replace(r"^\s*$", np.nan, regex=True)
                y = (
                    df[target_col].astype(str).str.strip().str.lower()
                    .map({pos_label.strip().lower(): 1}).fillna(0).astype(int)
                )
                X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
                if X.shape[1] == 0:
                    st.error("No features left after dropping.")
                    st.stop()
 
                for c in X.columns:
                    if X[c].dtype == object:
                        try:
                            X[c] = pd.to_numeric(X[c])
                        except Exception:
                            pass
 
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
 
                num = X.select_dtypes(include=[np.number]).columns.tolist()
                cat = X.select_dtypes(exclude=[np.number]).columns.tolist()
 
                pre = ColumnTransformer([
                    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                      ("sc", StandardScaler())]), num),
                    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
                ])
 
                clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
                pipe = Pipeline([("pre", pre), ("clf", clf)])
                pipe.fit(X_tr, y_tr)
 
                y_pred = pipe.predict(X_te)
                y_proba = pipe.predict_proba(X_te)[:, 1]
 
                metrics = {
                    "accuracy": float(accuracy_score(y_te, y_pred)),
                    "precision": float(precision_score(y_te, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_te, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_te, y_pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y_te, y_proba)),
                }
                st.json(metrics)
 
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(y_te, y_pred, ax=ax)
                st.pyplot(fig)
 
                Path("models").mkdir(exist_ok=True)
                joblib.dump(pipe, "models/auto_model.joblib")
                st.success("Saved: models/auto_model.joblib")
                load_model.clear()
 
                st.session_state.ACTIVE_MODEL_PATH = "models/auto_model.joblib"
                st.success("Switched to the newly trained model.")
                st.rerun()
 
            except Exception as e:
                st.error(f"Training failed: {e}")
 
 
# ----------------------------
# Tab 4: Explain Prediction (SHAP)
# ----------------------------
with tab4:
    st.subheader("Explain a prediction with SHAP")
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows which features drove the model's "
        "prediction for this specific customer — and by how much."
    )
 
    example_explain = (
        '{\n'
        '  "gender": "Female", "SeniorCitizen": 0,\n'
        '  "Partner": "Yes", "Dependents": "No",\n'
        '  "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",\n'
        '  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",\n'
        '  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",\n'
        '  "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",\n'
        '  "MonthlyCharges": 80.5, "TotalCharges": 970.2\n'
        '}'
    )
 
    json_shap = st.text_area("JSON input (same format as Single Prediction)", value=example_explain, height=260, key="shap_input")
    top_n = st.slider("Number of top features to show", min_value=5, max_value=20, value=10)
 
    if st.button("Explain this prediction"):
        try:
            record = json.loads(json_shap)
            df_single = pd.DataFrame([record])
 
            # Get prediction first
            res = predict_df(df_single, PIPE, THRESHOLD)
            prob = float(res.loc[0, "churn_proba"])
            pred = int(res.loc[0, "churn_pred"])
 
            verdict = "HIGH churn risk 🔴" if pred == 1 else "LOW churn risk 🟢"
            st.metric("Churn probability", f"{prob:.1%}", verdict)
 
            # Build SHAP explainer (cached)
            with st.spinner("Building explanation..."):
                explainer = load_shap_explainer(st.session_state.ACTIVE_MODEL_PATH)
                df_shap = explain_single(PIPE, explainer, df_single, top_n=top_n)
 
            st.subheader("What drove this prediction?")
 
            # Waterfall bar chart
            avg_churn_rate = 0.265  # Telco dataset baseline churn rate

            fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.5)))

            colors = ["#E8593C" if v > 0 else "#3B8BD4" for v in df_shap["shap_value"]]
            y_positions = range(len(df_shap))

            bars = ax.barh(
                list(y_positions),
                df_shap["shap_value"][::-1].values,
                color=colors[::-1],
                edgecolor="none",
                height=0.55,
            )

            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axvline(
                0, color="#F2A623", linewidth=1.5, linestyle=":",
                label=f"Baseline (avg churn rate: {avg_churn_rate:.1%})"
            )

            for bar, val in zip(bars, df_shap["shap_value"][::-1].values):
                x_pos = val
                ha = "left" if val >= 0 else "right"
                pad = 0.008 if val >= 0 else -0.008
                ax.text(
                    x_pos + pad,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}",
                    va="center",
                    ha=ha,
                    fontsize=9,
                    color="white" if abs(val) > 0.15 else ("black" if val >= 0 else "white"),
                )

            ax.set_yticks(list(y_positions))
            ax.set_yticklabels(df_shap["feature_clean"][::-1].values, fontsize=10)
            ax.set_xlabel("SHAP value (impact on churn probability)", fontsize=10)
            ax.set_title(f"Top {top_n} features driving this prediction", fontsize=11, pad=12)
            ax.legend(loc="lower right", fontsize=9)

            x_min, x_max = ax.get_xlim()
            ax.set_xlim(x_min - 0.06, x_max + 0.09)

            plt.tight_layout()
            st.pyplot(fig)
            # Legend
            st.caption("🔴 Red bars = feature increases churn risk   |   🔵 Blue bars = feature reduces churn risk")
 
            # Table view
            with st.expander("Show full SHAP table"):
                st.dataframe(
                    df_shap[["feature_clean", "shap_value", "direction"]]
                    .rename(columns={"feature_clean": "feature", "shap_value": "SHAP value"}),
                    use_container_width=True
                )
 
        except Exception as e:
            st.error(f"Explanation failed: {e}")
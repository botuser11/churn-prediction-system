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
    # 1) Normalize blank strings to NaN
    df_clean = df_clean.replace(r"^\s*$", np.nan, regex=True)

    # 2) Coerce numeric columns
    for c in num_cols:
        if c in df_clean.columns:
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
    
    schema_warnings(df_clean, expected)

    # Align for model
    df_for_model = align_columns(df_clean, expected)

    proba = pipe.predict_proba(df_for_model)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = pred
    return out


# ----------------------------
# Sidebar: Threshold modes + Metrics + Drivers
# ----------------------------
PIPE = load_model(st.session_state.ACTIVE_MODEL_PATH)

with st.sidebar:
    st.markdown(f"**Active model:** `{st.session_state.ACTIVE_MODEL_PATH}`")

    st.subheader("Decision Threshold")

    modes = ["Manual", "F1 (Validation-tuned)", "Cost (Min expected cost)"]
    default_mode = 0
    st.session_state.THRESH_MODE = st.selectbox("Threshold mode", modes, index=default_mode)

    # thresholds from artifacts (if available)
    f1_thr = None
    if META.get("phase2_test_metrics") and "threshold" in META["phase2_test_metrics"]:
        f1_thr = float(META["phase2_test_metrics"]["threshold"])

    cost_thr = None
    if META.get("cost_threshold") and "best_threshold_by_cost" in META["cost_threshold"]:
        cost_thr = float(META["cost_threshold"]["best_threshold_by_cost"]["threshold"])

    # --- Make sure we keep two values:
    # 1) MANUAL_THRESHOLD: what the user last set with the slider
    # 2) ACTIVE_THRESHOLD: the threshold actually used for predictions (manual/F1/cost)

    if "MANUAL_THRESHOLD" not in st.session_state:
        st.session_state.MANUAL_THRESHOLD = float(CFG.get("threshold", 0.5))

    if "ACTIVE_THRESHOLD" not in st.session_state:
        st.session_state.ACTIVE_THRESHOLD = float(st.session_state.MANUAL_THRESHOLD)

    if st.session_state.THRESH_MODE == "Manual":
        st.session_state.MANUAL_THRESHOLD = st.slider(
            "Flag as positive when probability ≥",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state.MANUAL_THRESHOLD),
            step=0.01,
        )
        st.session_state.ACTIVE_THRESHOLD = float(st.session_state.MANUAL_THRESHOLD)

    elif st.session_state.THRESH_MODE == "F1 (Validation-tuned)":
        if f1_thr is None:
            st.warning("No Phase 2 tuned threshold found. Run Phase 2 first.")
            st.session_state.ACTIVE_THRESHOLD = float(st.session_state.MANUAL_THRESHOLD)
        else:
            st.session_state.ACTIVE_THRESHOLD = float(f1_thr)
            st.info(f"Using F1-tuned threshold: {float(f1_thr):.3f}")

    else:  # Cost
        if cost_thr is None:
            st.warning("No cost-threshold found. Run Phase 3 cost_threshold.py first.")
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
        st.caption("Run Phase 2/3 training to populate metrics in reports/train_final_phase2.json")

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

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch CSV Prediction", "Train New Model"])


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
                    ["<none>"] + available_cols,
                    index=0,
                    help="Pick a column that uniquely identifies each row, if you have one (e.g., customerID)."
                )
                top_k = st.slider(
                    "Show top N highest-risk rows",
                    min_value=5,
                    max_value=200,
                    value=20,
                    help="Sorted by probability, descending"
                )
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

                    st.write("Detailed view (first 200 rows, sorted by probability):")
                    st.dataframe(
                        res.sort_values("churn_proba", ascending=False)
                        .loc[:, adv_cols]
                        .head(200),
                        use_container_width=True
                    )

                safe_mode = st.session_state.THRESH_MODE.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                fname = f"predictions_{Path(st.session_state.ACTIVE_MODEL_PATH).stem}_{safe_mode}_thr{THRESHOLD:.3f}.csv"

                st.download_button(
                    f"Download {fname}",
                    res.to_csv(index=False),
                    fname,
                    "text/csv"
)


            except Exception as e:
                st.error(f"Failed: {e}")


# ----------------------------
# Tab 3: Train new model (on uploaded CSV)
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
                # Normalize blanks
                df = df.replace(r"^\s*$", np.nan, regex=True)

                y = (
                    df[target_col].astype(str).str.strip().str.lower()
                    .map({pos_label.strip().lower(): 1}).fillna(0).astype(int)
                )
                X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
                if X.shape[1] == 0:
                    st.error("No features left after dropping. Please adjust 'Columns to drop'.")
                    st.stop()

                # best-effort numeric coercion
                for c in X.columns:
                    if X[c].dtype == object:
                        try:
                            X[c] = pd.to_numeric(X[c])
                        except Exception:
                            pass

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )

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

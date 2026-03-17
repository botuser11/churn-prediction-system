# Customer Churn Prediction System

An end-to-end machine learning application for predicting customer churn in the telecom industry. Built with Python, scikit-learn, and Streamlit — combining a production-style ML pipeline with an interactive UI suitable for both technical and non-technical users.

**Live demo:** churn-prediction-system-dffqbps5hjvigpbjqukw3k.streamlit.app

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC (test) | **0.843** |
| PR-AUC (test) | **0.633** |
| Brier Score (test) | **0.168** |
| Train / Val / Test | 4225 / 1409 / 1409 |

> Model: Logistic Regression with class balancing, trained on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

## Key findings

**Top features driving churn risk:**
- Month-to-month contract — strongest positive predictor (coef: +0.74)
- Fiber optic internet service (coef: +0.59)
- Electronic check payment method (coef: +0.33)

**Top features reducing churn risk:**
- Tenure — strongest retention signal (coef: −1.21)
- Two-year contract (coef: −0.86)
- DSL internet service (coef: −0.54)

---

## Features

### Single prediction
Paste a JSON customer record and get an instant churn probability with binary classification at a configurable threshold.

### Batch CSV prediction
Upload a CSV of customers and get probability scores for every row, a distribution plot, top at-risk customers sorted by risk, and a downloadable results file.

### Explain prediction (SHAP)
Per-customer explainability using SHAP (SHapley Additive exPlanations). A waterfall bar chart shows exactly which features drove each individual prediction up or down and by how much — making predictions interpretable for business users.

### Model training
Upload any binary classification CSV and train a new logistic regression pipeline on the fly. Evaluation metrics and confusion matrix are shown immediately, and the new model becomes the active model automatically.

### Threshold control
Three modes: manual slider, F1-tuned threshold (optimised on validation set), and cost-optimal threshold (minimises expected business cost of false positives vs false negatives).

---

## Project structure

```
churn/
├── app.py                        # Streamlit UI — all four tabs
├── config.json                   # Data path, target column, model path, threshold
├── data/
│   └── raw/
│       └── telco_churn.csv
├── models/
│   └── final_model.joblib        # Trained sklearn pipeline (pre + clf)
├── reports/
│   ├── train_final_phase2.json   # ROC-AUC, PR-AUC, Brier, split sizes
│   ├── final_metrics_phase2.json # F1-tuned threshold + test metrics
│   ├── threshold_by_cost.json    # Cost-optimal threshold
│   ├── top_positive_drivers.csv  # Top churn-increasing features
│   ├── top_negative_drivers.csv  # Top churn-reducing features
│   └── probas_lr.npz             # Val/test probabilities for threshold tuning
└── src/
    ├── model_metadata.py         # Loads report JSONs for sidebar display
    ├── shap_explainer.py         # SHAP LinearExplainer + explain_single()
    ├── pipeline_factory.py       # Builds sklearn preprocessing + LR pipeline
    ├── preprocess.py             # Data cleaning for Telco dataset
    ├── split_utils.py            # Deterministic train/val/test splits
    ├── train_final.py            # Phase 1 training — saves model + probas
    ├── tune_threshold.py         # Phase 2 — F1-optimal threshold on val set
    ├── cost_threshold.py         # Phase 3 — cost-based threshold optimisation
    └── feature_importance_lr.py  # Extracts + saves LR coefficients
```

---

## Reproducing the results

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the training pipeline in order**
```bash
python src/train_final.py
python src/tune_threshold.py
python src/cost_threshold.py
python src/feature_importance_lr.py
```

**3. Launch the app**
```bash
streamlit run app.py
```

---

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn, SHAP |
| Data | pandas, NumPy |
| UI | Streamlit |
| Visualisation | matplotlib |
| Serialisation | joblib |

---

## Pipeline design

```
ColumnTransformer (preprocessing)
    ├── Numerical → MedianImputer → StandardScaler
    └── Categorical → MostFrequentImputer → OneHotEncoder
                        ↓
            LogisticRegression (class_weight="balanced")
```

Consistent preprocessing between training and inference with no data leakage, serialised cleanly via joblib.

---

## What I would add next

- Gradient Boosting model comparison (XGBoost vs LR on ROC-AUC, PR-AUC, F1)
- Calibration plot to validate probability estimates
- FastAPI wrapper for production REST API
- Docker container for reproducible deployment

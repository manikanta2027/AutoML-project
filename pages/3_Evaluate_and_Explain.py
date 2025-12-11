import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.evaluation import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_pred_vs_actual,
    plot_residuals,
)
from utils.shap_engine import compute_shap_and_save

# -------------------------
# Session state defaults
# -------------------------
defaults = {
    "df": None,
    "eda_summary": None,
    "cleaned_df": None,
    "target": None,
    "task_type": None,

    "models": None,
    "best_model_name": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "X_columns": None,

    "cv_scores": None,
    "test_scores": None,

    "last_scores": None,
    "last_image_paths": None,
    "last_model_name": None,
    "selected_model_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("📈 Evaluation & 🔍 Explainability (SHAP)")

# -------------------------------------------------------------------
# 0) SAFETY CHECKS
# -------------------------------------------------------------------
required_keys = [
    "models", "X_train", "X_test", "y_train", "y_test",
    "target", "task_type", "test_scores"
]
missing = [k for k in required_keys if st.session_state.get(k) is None]

if missing:
    st.error(
        "❌ Missing training data or models.\n\n"
        "Please train models in **🚀 Train Models** first."
    )
    st.stop()

models = st.session_state.models
X_train = st.session_state.X_train
X_test = st.session_state.X_test
y_train = st.session_state.y_train
y_test = st.session_state.y_test
task_type = st.session_state.task_type
target = st.session_state.target
test_scores = st.session_state.test_scores

# -------------------------------------------------------------------
# 1) Choose model
# -------------------------------------------------------------------
keys = list(models.keys())
idx = 0
if st.session_state.best_model_name in keys:
    idx = keys.index(st.session_state.best_model_name)

model_name = st.selectbox("Choose model to evaluate", keys, index=idx)
model = models[model_name]

# Optional: button to see code from here
if st.button("💻 View Code for This Model"):
    st.session_state.selected_model_name = model_name
    st.switch_page("pages/6_Model_Code.py")

# Predict on TEST set
try:
    y_pred = model.predict(X_test)
except Exception as e:
    st.error("Prediction on test set failed: " + str(e))
    y_pred = None

image_paths = {}
scores = {}

# -------------------------------------------------------------------
# 2) Classification evaluation
# -------------------------------------------------------------------
if task_type in ["Binary Classification", "Multi-Class Classification"]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    st.subheader("📊 Classification Metrics (TEST set)")

    if y_pred is not None:
        try:
            scores = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision (weighted)": precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "Recall (weighted)": recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "F1 (weighted)": f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
            }
        except Exception as e:
            st.warning("Could not compute some classification metrics: " + str(e))

    for k, v in scores.items():
        if isinstance(v, (int, float, np.floating)):
            st.write(f"- **{k}**: {v:.4f}")
        else:
            st.write(f"- **{k}**: {v}")

    # Confusion matrix
    cm_path = f"reports/{model_name}_cm_test.png"
    try:
        plot_confusion_matrix(model, X_test, y_test, cm_path)
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix")
            image_paths["Confusion Matrix (test)"] = cm_path
    except Exception as e:
        st.info("Confusion matrix failed: " + str(e))

    # ROC / PR for *binary* classification
    if task_type == "Binary Classification":
        roc_path = f"reports/{model_name}_roc_test.png"
        pr_path = f"reports/{model_name}_pr_test.png"
        try:
            if hasattr(model, "predict_proba"):
                plot_roc_curve(model, X_test, y_test, roc_path)
                plot_precision_recall_curve(model, X_test, y_test, pr_path)

                if os.path.exists(roc_path):
                    st.image(roc_path, caption="ROC Curve")
                    image_paths["ROC (test)"] = roc_path
                if os.path.exists(pr_path):
                    st.image(pr_path, caption="Precision-Recall Curve")
                    image_paths["PR (test)"] = pr_path
            else:
                st.info("Model does not support predict_proba – ROC/PR skipped.")
        except Exception as e:
            st.info("ROC/PR plotting failed: " + str(e))

    # Feature importance
    fi_path = f"reports/{model_name}_fi.png"
    try:
        if hasattr(model, "feature_importances_") or hasattr(model, "coef_") or hasattr(
            getattr(model, "named_steps", {}).get("model", None), "feature_importances_"
        ):
            plot_feature_importance(model, X_train.columns, fi_path)
            if os.path.exists(fi_path):
                st.image(fi_path, caption="Feature Importance")
                image_paths["Feature Importance"] = fi_path
        else:
            st.info("Feature importance not available for this model.")
    except Exception:
        st.info("Feature importance not available for this model.")

# -------------------------------------------------------------------
# 3) Regression evaluation
# -------------------------------------------------------------------
else:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    st.subheader("📊 Regression Metrics (TEST set)")

    if y_pred is not None:
        try:
            scores = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
            }
        except Exception as e:
            st.warning("Could not compute regression metrics: " + str(e))

    for k, v in scores.items():
        if isinstance(v, (int, float, np.floating)):
            st.write(f"- **{k}**: {v:.4f}")
        else:
            st.write(f"- **{k}**: {v}")

    # Pred vs Actual + residuals
    pred_path = f"reports/{model_name}_pred_test.png"
    res_path = f"reports/{model_name}_res_test.png"
    try:
        plot_pred_vs_actual(y_test, y_pred, pred_path)
        plot_residuals(y_test, y_pred, res_path)
        if os.path.exists(pred_path):
            st.image(pred_path, caption="Predicted vs Actual")
            image_paths["Pred vs Actual (test)"] = pred_path
        if os.path.exists(res_path):
            st.image(res_path, caption="Residual Plot")
            image_paths["Residuals (test)"] = res_path
    except Exception as e:
        st.info("Regression plots failed: " + str(e))

    # Feature importance
    fi_path = f"reports/{model_name}_fi.png"
    try:
        if hasattr(model, "feature_importances_") or hasattr(model, "coef_") or hasattr(
            getattr(model, "named_steps", {}).get("model", None), "feature_importances_"
        ):
            plot_feature_importance(model, X_train.columns, fi_path)
            if os.path.exists(fi_path):
                st.image(fi_path, caption="Feature Importance")
                image_paths["Feature Importance"] = fi_path
    except Exception:
        pass

# -------------------------------------------------------------------
# 4) Leaderboard
# -------------------------------------------------------------------
st.subheader("🏆 Leaderboard (Test Scores)")

ts = st.session_state.test_scores or {}
if ts:
    df_leader = pd.DataFrame(
        [
            (k, pd.to_numeric(v, errors="coerce"))
            for k, v in ts.items()
        ],
        columns=["Model", "Test Score"],
    )
    df_leader["Test Score"] = df_leader["Test Score"].fillna(0.0)

    ascending = (task_type == "Regression")
    df_leader = df_leader.sort_values("Test Score", ascending=ascending)

    st.dataframe(df_leader, use_container_width=True)

    fig = go.Figure(
        go.Bar(
            x=df_leader["Model"],
            y=df_leader["Test Score"],
            text=df_leader["Test Score"].round(3),
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Model Leaderboard (Test Set)",
        xaxis_title="Model",
        yaxis_title="Test Score",
        
    )
    fig.update_layout(
    xaxis_tickangle=-45,   
    margin=dict(b=150),    
)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No test scores found. Train models first.")

# Save for report page (even if SHAP not run)
st.session_state.last_scores = scores
st.session_state.last_image_paths = image_paths
st.session_state.last_model_name = model_name

# -------------------------------------------------------------------
# 5) SHAP Explainability (with guard for big datasets)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 5) SHAP Explainability
# -------------------------------------------------------------------
st.subheader("🔍 SHAP Explainability (tree models only, sampled)")

# Small helper: detect inner estimator type
def _get_inner_estimator(m):
    if hasattr(m, "steps"):
        return m.steps[-1][1]
    return m

inner = _get_inner_estimator(model)
inner_name = inner.__class__.__name__

n_rows, n_features = X_test.shape

st.caption(
    f"Current model: **{inner_name}**, Test shape: "
    f"{n_rows} rows × {n_features} features"
)

# Allow SHAP ONLY for tree-based models, and only for moderate size
tree_keywords = ["Forest", "Tree", "Boost", "XGB", "HistGradient"]
is_tree = any(k.lower() in inner_name.lower() for k in tree_keywords)

if not is_tree:
    st.info(
        "⚠️ SHAP is enabled only for **tree-based models** "
        "(RandomForest, GradientBoosting, XGBoost, HistGradientBoosting).\n\n"
        "Select a tree model above to view SHAP plots."
    )
elif n_rows > 3000 or n_features > 120:
    st.info(
        f"⚠️ SHAP disabled for this dataset size "
        f"({n_rows} rows, {n_features} features).\n\n"
        "Run SHAP on a smaller dataset or with fewer features."
    )
else:
    if st.button("Compute SHAP explanations"):
        with st.spinner("Computing SHAP on a sampled test subset..."):
            try:
                shap_saved, shap_ctx = compute_shap_and_save(
                    model,
                    X_test,
                    sample_frac=0.10,   # 10% of test set
                    max_samples=400,    # at most 400 rows
                    random_state=42,
                    top_k_dependence=6,
                    out_dir="reports/shap",
                )

                any_image = False

                if shap_saved.get("summary") and os.path.exists(shap_saved["summary"]):
                    st.image(shap_saved["summary"], caption="SHAP Summary Plot", width=650)
                    image_paths["SHAP Summary"] = shap_saved["summary"]
                    any_image = True

                if shap_saved.get("bar") and os.path.exists(shap_saved["bar"]):
                    st.image(
                        shap_saved["bar"], caption="SHAP Feature Importance", width=650
                    )
                    image_paths["SHAP Importance"] = shap_saved["bar"]
                    any_image = True

                if shap_saved.get("dependence"):
                    st.markdown("#### SHAP Dependence Plots")
                    for p in shap_saved["dependence"]:
                        if p and os.path.exists(p):
                            st.image(p, caption=os.path.basename(p), width=580)
                            image_paths[f"SHAP Dependence {p}"] = p
                            any_image = True

                if not any_image:
                    if shap_ctx.get("error"):
                        st.warning(
                            "SHAP did not produce plots. Reason:\n\n"
                            f"`{shap_ctx['error']}`"
                        )
                    else:
                        st.warning(
                            "SHAP did not produce any images. "
                            "The model type or SHAP backend might not support this."
                        )

                # update session-state for report page
                st.session_state.last_scores = scores
                st.session_state.last_image_paths = image_paths
                st.session_state.last_model_name = model_name

            except Exception as e:
                st.error("SHAP computation failed: " + str(e))


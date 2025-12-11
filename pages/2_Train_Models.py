import streamlit as st
import pandas as pd
import numpy as np

from engines.regression_engine import train_regression_models
from engines.classification_engine import train_classification_models
from utils import eda  # for SMOTE availability flag etc.

SEED = 42
TEST_SIZE = 0.2

# -------------------------
# Session state defaults
# -------------------------
defaults = {
    "df": None,
    "eda_summary": None,
    "cleaned_df": None,
    "target": None,
    "task_type": None,

    # Training
    "models": None,
    "best_model_name": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "X_columns": None,
    "cv_scores": None,
    "test_scores": None,

    # Eval / report / code
    "last_scores": None,
    "last_image_paths": None,
    "last_model_name": None,
    "selected_model_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🚀 Train Models")

# -------------------------
# Guard: need cleaned_df
# -------------------------
if st.session_state.cleaned_df is None:
    st.error("Please upload data and run cleaning in **📊 EDA & Cleaning** first.")
    st.stop()

df = st.session_state.cleaned_df

# -------------------------
# Target & task type
# -------------------------
if st.session_state.target in df.columns:
    default_idx = list(df.columns).index(st.session_state.target)
else:
    default_idx = 0

target = st.selectbox("Select target column", df.columns, index=default_idx)
st.session_state.target = target

task_type = st.selectbox(
    "Task Type",
    ["Regression", "Binary Classification", "Multi-Class Classification"],
    index=["Regression", "Binary Classification", "Multi-Class Classification"].index(
        st.session_state.task_type
    ) if st.session_state.task_type in ["Regression", "Binary Classification", "Multi-Class Classification"] else 0
)
st.session_state.task_type = task_type

# -------------------------
# SMOTE (classification only)
# -------------------------
apply_smote = False
if task_type != "Regression":
    apply_smote = st.checkbox("Apply SMOTE on training set (if available)", value=False)
    if apply_smote and not getattr(eda, "_HAS_SMOTE", False):
        st.warning(
            "imbalanced-learn/SMOTE not available — SMOTE will be ignored."
        )
        apply_smote = False

# -------------------------
# Train button
# -------------------------
if st.button("🚀 Train models (on cleaned dataset)"):
    train_df = df.copy()

    with st.spinner("Training models..."):
        if task_type == "Regression":
            out = train_regression_models(
                train_df, target, seed=SEED, test_size=TEST_SIZE
            )
        else:
            out = train_classification_models(
                train_df,
                target,
                seed=SEED,
                test_size=TEST_SIZE,
                apply_smote=apply_smote,
            )

    try:
        (
            cv_scores,
            test_scores,
            models,
            X_train,
            X_test,
            y_train,
            y_test,
        ) = out
    except Exception:
        st.error(
            "train_*_models must return: "
            "(cv_scores, test_scores, models, X_train, X_test, y_train, y_test)"
        )
        st.stop()

    # Save to session_state
    st.session_state.models = models
    st.session_state.cv_scores = cv_scores
    st.session_state.test_scores = test_scores
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.X_columns = list(X_train.columns)

    # pick best model on numeric test score
    numeric_scores = {
        k: v
        for k, v in test_scores.items()
        if isinstance(v, (int, float, np.floating))
    }
    if numeric_scores:
        best_name = max(numeric_scores, key=numeric_scores.get)
    else:
        best_name = list(test_scores.keys())[0] if test_scores else None

    st.session_state.best_model_name = best_name
    st.success(f"✅ Training finished — best model: **{best_name}**")

# -------------------------
# Always show current results (persist across page switches)
# -------------------------
if st.session_state.cv_scores:
    st.subheader("📊 CV (Train) Scores")
    st.dataframe(
        pd.DataFrame(
            list(st.session_state.cv_scores.items()),
            columns=["Model", "CV Score"],
        ).set_index("Model")
    )

if st.session_state.test_scores:
    st.subheader("📊 Test Scores")
    st.dataframe(
        pd.DataFrame(
            list(st.session_state.test_scores.items()),
            columns=["Model", "Test Score"],
        ).set_index("Model")
    )

    # Simple leaderboard here too (optional)
    ts = st.session_state.test_scores
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
    st.subheader("🏆 Leaderboard (Test Scores)")
    st.dataframe(df_leader, use_container_width=True)

# -------------------------
# 💻 Model Code button
# -------------------------
st.markdown("---")
st.subheader("💻 Get Code for a Trained Model")

if st.session_state.models:
    model_keys = list(st.session_state.models.keys())

    # Default: best model if exists
    if st.session_state.best_model_name in model_keys:
        default_idx = model_keys.index(st.session_state.best_model_name)
    else:
        default_idx = 0

    code_model_name = st.selectbox(
        "Select model for code snippet",
        model_keys,
        index=default_idx,
        key="code_model_select_train",
    )

    if st.button("📄 View Code for This Model"):
        st.session_state.selected_model_name = code_model_name
        st.switch_page("pages/6_Model_Code.py")
else:
    st.info("Train at least one model to enable the Model Code page.")

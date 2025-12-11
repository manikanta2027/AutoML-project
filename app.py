# app.py – main entry for multi-page AutoML

import streamlit as st

import streamlit as st

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "target" not in st.session_state:
    st.session_state.target = None

if "X_columns" not in st.session_state:
    st.session_state.X_columns = None

if "models" not in st.session_state:
    st.session_state.models = {}

if "best_model" not in st.session_state:
    st.session_state.best_model = None

if "cv_scores" not in st.session_state:
    st.session_state.cv_scores = None

if "test_scores" not in st.session_state:
    st.session_state.test_scores = None

if "X_train" not in st.session_state:
    st.session_state.X_train = None

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_train" not in st.session_state:
    st.session_state.y_train = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None


st.set_page_config(page_title="AutoML System", layout="wide")

st.title("🤖 AutoML System — Multi Page Edition")

st.markdown(
    """
Welcome to your **AutoML platform**.

Use the sidebar to navigate:

1. **EDA & Cleaning** – Upload dataset, run EDA, auto-clean  
2. **Train Models** – Train regression/classification models  
3. **Evaluate & Explain** – Metrics, plots, SHAP explanations  
4. **Predict on New Data** – Template CSV + single-row predictions  
5. **Generate Report** – Export PDF report (using your existing generator)
"""
)

st.info("Start on **📊 EDA & Cleaning** after creating the `pages/` folder and files.")

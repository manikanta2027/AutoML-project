import os
import json
import zipfile
from io import BytesIO
import streamlit as st
import pandas as pd
from datetime import datetime

# ======================================
# REQUIRED SESSION DATA
# ======================================
required = ["models", "best_model_name", "cleaned_df", "target", "X_columns"]
missing = [k for k in required if k not in st.session_state or st.session_state[k] is None]

st.title("💻 Export Model Code & Templates")

if missing:
    st.error(
        "❌ Cannot export code.\n"
        "Train a model first, then return here."
    )
    st.stop()

models = st.session_state.models
best_model_name = st.session_state.best_model_name
model = models[best_model_name]
df = st.session_state.cleaned_df
target = st.session_state.target
features = st.session_state.X_columns

numeric_features = df[features].select_dtypes(include=["number"]).columns.tolist()
categorical_features = df[features].select_dtypes(exclude=["number"]).columns.tolist()

# ======================================
# HELPER: JOIN LINES SAFELY
# ======================================
def build(*lines):
    return "\n".join(lines)

# ======================================
# TRAINING SCRIPT
# ======================================
def generate_training_script():
    lines = [
        "# ==========================================================",
        "# AUTO-GENERATED TRAINING TEMPLATE (Edit before use)",
        f"# Model: {best_model_name}",
        f"# Target: {target}",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "# ==========================================================",
        "",
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.linear_model import Ridge   # TODO: change model",
        "import joblib",
        "",
        "# Load dataset",
        "df = pd.read_csv('your_dataset.csv')  # TODO update file",
        f"target = '{target}'",
        "X = df.drop(columns=[target])",
        "y = df[target]",
        "",
        f"numeric_features = {numeric_features}",
        f"categorical_features = {categorical_features}",
        "",
        "# Preprocessing",
        "numeric_transformer = Pipeline([",
        "    ('imputer', SimpleImputer(strategy='mean')),",
        "    ('scaler', StandardScaler())",
        "])",
        "",
        "categorical_transformer = Pipeline([",
        "    ('imputer', SimpleImputer(strategy='most_frequent')),",
        "    ('onehot', OneHotEncoder(handle_unknown='ignore'))",
        "])",
        "",
        "preprocess = ColumnTransformer([",
        "    ('num', numeric_transformer, numeric_features),",
        "    ('cat', categorical_transformer, categorical_features)",
        "])",
        "",
        "# Model",
        "model = Ridge()  # TODO replace model",
        "",
        "# Full pipeline",
        "pipeline = Pipeline([",
        "    ('preprocess', preprocess),",
        "    ('model', model)",
        "])",
        "",
        "# Split",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, random_state=42",
        ")",
        "",
        "# Train",
        "pipeline.fit(X_train, y_train)",
        "",
        "print('Train Score:', pipeline.score(X_train, y_train))",
        "print('Test Score:', pipeline.score(X_test, y_test))",
        "",
        "# Save",
        "joblib.dump(pipeline, 'model.pkl')",
        "print('Model saved!')"
    ]
    return build(*lines)

# ======================================
# PREDICTION SCRIPT
# ======================================
def generate_prediction_script():
    sample = {col: "VALUE" for col in features}

    lines = [
        "import joblib",
        "import pandas as pd",
        "",
        "model = joblib.load('model.pkl')",
        "",
        "def predict(input_dict):",
        "    df = pd.DataFrame([input_dict])",
        "    return model.predict(df)[0]",
        "",
        f"example = {sample}",
        "print('Prediction:', predict(example))",
    ]
    return build(*lines)

# ======================================
# README
# ======================================
def generate_readme():
    lines = [
        "# AutoML Model Export",
        "",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Files:",
        "- train.py",
        "- predict.py",
        "- requirements.txt",
        "- notebook.ipynb",
        "",
        "## Train:",
        "python train.py",
        "",
        "## Predict:",
        "python predict.py",
    ]
    return build(*lines)

# ======================================
# REQUIREMENTS
# ======================================
def generate_requirements():
    return build(
        "scikit-learn==1.3.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "joblib==1.3.2",
    )

# ======================================
# NOTEBOOK (minimal)
# ======================================
def generate_notebook():
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "cells": [
            {"cell_type": "markdown", "source": f"# AutoML Notebook – {best_model_name}"},
            {"cell_type": "code", "source": "!pip install scikit-learn pandas numpy joblib"},
            {"cell_type": "code", "source": generate_training_script()},
        ],
    }

# ======================================
# ZIP PACKAGE
# ======================================
def build_zip_package():
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("train.py", generate_training_script())
        z.writestr("predict.py", generate_prediction_script())
        z.writestr("requirements.txt", generate_requirements())
        z.writestr("README.md", generate_readme())
        z.writestr("notebook.ipynb", json.dumps(generate_notebook(), indent=2))
    buffer.seek(0)
    return buffer

# ======================================
# UI
# ======================================
tab1, tab2, tab3 = st.tabs(["Training Script", "Prediction Script", "Export ZIP"])

with tab1:
    code = generate_training_script()
    st.code(code, language="python")
    st.download_button("Download train.py", code, "train.py")

with tab2:
    code = generate_prediction_script()
    st.code(code, language="python")
    st.download_button("Download predict.py", code, "predict.py")

with tab3:
    if st.button("Generate ZIP Export Package"):
        zipbuf = build_zip_package()
        st.download_button(
            "📦 Download model_export_package.zip",
            data=zipbuf,
            file_name="model_export_package.zip",
            mime="application/zip"
        )
        st.success("Exported successfully!")

import os
import streamlit as st
from utils.report_generator import generate_report  # your PDF generator


# ---- Initialize session state (prevents reset on page switch) ----
defaults = {
    "df": None,
    "eda_summary": None,
    "cleaned_df": None,
    "target": None,
    "task_type": None,

    # Training stage
    "models": None,
    "best_model_name": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "X_columns": None,

    # Evaluation / SHAP / Report
    "last_scores": None,
    "last_image_paths": None,
    "last_model_name": None,

    # Model code page
    "selected_model_name": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.title("📄 Generate PDF Report")

# -------------------------------------------------------------------
# Required Items From Previous Pages
# -------------------------------------------------------------------
required = [
    "last_model_name",
    "last_scores",
    "last_image_paths",
    "task_type",
    "target",
    "cleaned_df",
]

missing = [k for k in required if k not in st.session_state or st.session_state[k] is None]

if missing:
    st.error(
        "❌ Report inputs are missing.\n\n"
        "Please complete these steps first:\n"
        "1️⃣ Train models → **🚀 Train Models**\n"
        "2️⃣ Evaluate model + run SHAP → **📈 Evaluate & Explain**\n\n"
        "Then return to this page."
    )
    st.stop()

# -------------------------------------------------------------------
# Load Data From Session State
# -------------------------------------------------------------------
model_name = st.session_state.last_model_name
scores = st.session_state.last_scores
image_paths = dict(st.session_state.last_image_paths)  # copy safely
task_type = st.session_state.task_type
target = st.session_state.target
df_used = st.session_state.cleaned_df

# -------------------------------------------------------------------
# Display Basic Information
# -------------------------------------------------------------------
st.write(f"### 🧠 Model for report: **{model_name}**")
st.write(f"### 🏹 Task type: **{task_type}**")
st.write(f"### 🎯 Target column: **{target}**")
st.write(f"### 📊 Rows in cleaned dataset: **{df_used.shape[0]}**")

# -------------------------------------------------------------------
# Prepare Summary Text
# -------------------------------------------------------------------
steps_text = (
    f"AutoML report — model: {model_name}\n"
    f"Task type: {task_type}\n"
    f"Target: {target}\n"
    f"Rows (cleaned): {df_used.shape[0]}"
)

# -------------------------------------------------------------------
# Add EDA Images (if available)
# -------------------------------------------------------------------
eda_corr_path = "reports/eda/corr_heatmap.png"
if os.path.exists(eda_corr_path):
    image_paths["Correlation Heatmap"] = eda_corr_path

# -------------------------------------------------------------------
# Clean Dead / Missing Image Paths
# -------------------------------------------------------------------
valid_image_paths = {}
for label, img_path in image_paths.items():
    if isinstance(img_path, str) and os.path.exists(img_path):
        valid_image_paths[label] = img_path

image_paths = valid_image_paths

# Warn if no images found
if not image_paths:
    st.warning("⚠ No images found for report. The PDF will include only metrics and summary text.")

# -------------------------------------------------------------------
# Generate PDF Button
# -------------------------------------------------------------------
if st.button("📄 Generate PDF Report"):
    try:
        pdf_path = generate_report(
            task_type=task_type,
            model_name=model_name,
            scores=scores,
            image_paths=image_paths,
            steps_text=steps_text,
        )

        st.success("✅ PDF report successfully created!")

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download AutoML Report",
                data=f,
                file_name="AutoML_Report.pdf",
                mime="application/pdf",
            )

    except Exception as e:
        st.error(f"❌ PDF generation failed:\n\n{e}")

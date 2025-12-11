import io
import streamlit as st
import pandas as pd

st.title("🧪 Predict on New Data (CSV)")

# -----------------------------
# SAFETY CHECKS
# -----------------------------
required = ["models", "best_model_name", "cleaned_df", "target", "X_columns"]
missing = [k for k in required if k not in st.session_state or st.session_state[k] is None]

if missing:
    st.error(
        "❌ Missing model or cleaned dataset.\n\n"
        "Please train models in **🚀 Train Models** first."
    )
    st.stop()

models = st.session_state.models
best_model_name = st.session_state.best_model_name
model = models[best_model_name]
cleaned_df = st.session_state.cleaned_df
target = st.session_state.target
feature_cols = st.session_state.X_columns

st.markdown(f"### 🧠 Using Best Model: **{best_model_name}**")

# -----------------------------
# STORE PREDICTION HISTORY IN SESSION
# -----------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# -----------------------------
# TEMPLATE DOWNLOAD
# -----------------------------
st.subheader("📄 Download Input Template")

template_df = cleaned_df.drop(columns=[target], errors="ignore")
sample_row = template_df.sample(1, random_state=42)

buffer = io.StringIO()
sample_row.to_csv(buffer, index=False)

st.download_button(
    "📥 Download Template CSV",
    buffer.getvalue(),
    file_name="prediction_template.csv",
    mime="text/csv",
)

st.info("Edit the downloaded CSV and upload it below.")

# -----------------------------
# UPLOAD FOR PREDICTION
# -----------------------------
st.subheader("🧪 Upload CSV for Prediction")

pred_file = st.file_uploader("Upload CSV with same columns", type=["csv"])

if pred_file:
    try:
        user_df = pd.read_csv(pred_file)
        st.write("### 📌 Uploaded Data")
        st.dataframe(user_df)

        # --- COLUMN CHECKS ---
        missing = set(feature_cols) - set(user_df.columns)
        extra = set(user_df.columns) - set(feature_cols)

        if missing:
            st.error(f"❌ Missing required columns: {list(missing)}")
            st.stop()

        if extra:
            st.warning(f"⚠ Extra columns ignored: {list(extra)}")
            user_df = user_df[feature_cols]

        # --- Predict ---
        pred_value = model.predict(user_df)[0]

        st.success(f"### ✅ Prediction: **{pred_value}**")

        # Probability if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_df)[0]
            st.write("### Class Probabilities")
            st.write({cls: float(p) for cls, p in zip(model.classes_, proba)})

        # Save result to history
        st.session_state.prediction_history.append(
            {"input": user_df.to_dict(), "prediction": pred_value}
        )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# -----------------------------
# PREDICTION HISTORY
# -----------------------------
if st.session_state.prediction_history:
    st.subheader("📝 Previous Predictions")
    for i, entry in enumerate(st.session_state.prediction_history):
        st.write(f"**Prediction {i+1}:** {entry['prediction']}")

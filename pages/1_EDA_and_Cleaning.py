import os
import streamlit as st
import pandas as pd
from utils import eda   # correct import

# =====================================================
# SESSION STATE INITIALIZATION  (DO NOT REMOVE)
# =====================================================
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



def safe_path(p):
    return p if p and os.path.exists(p) else None


# =====================================================
# PAGE TITLE
# =====================================================
st.title("📊 EDA & 🧹 Cleaning")


# =====================================================
# FILE UPLOAD — persists across pages
# =====================================================
file = st.file_uploader("Upload CSV dataset", type=["csv"], key="eda_upload")

if file:
    # Store dataset in session_state (IMPORTANT)
    st.session_state.df = pd.read_csv(file)
    st.success("✅ Dataset loaded")

# ----- SHOW DATASET IF AVAILABLE -----
if st.session_state.df is not None:
    df = st.session_state.df

    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head())

    # =====================================================
    # RUN EDA (only first time OR when dataset changes)
    # =====================================================
    if st.session_state.eda_summary is None:
        with st.spinner("Running EDA..."):
            st.session_state.eda_summary = eda.run_eda(
                df, out_dir="reports/eda"
            )

    eda_summary = st.session_state.eda_summary

    # ----------------- Basic EDA -----------------
    st.subheader("🔎 Quick EDA Summary")

    st.write(
        f"- Rows: **{eda_summary.get('rows','N/A')}**  |  "
        f"Columns: **{eda_summary.get('columns','N/A')}**"
    )

    # Missing stats table
    if "missing" in eda_summary:
        missing_dict = eda_summary["missing"] or {}
        rows = []
        for col, stats in missing_dict.items():
            rows.append({
                "column": col,
                "dtype": eda_summary["column_types"].get(col, "unknown"),
                "missing_count": stats.get("missing_count", 0),
                "missing_pct": stats.get("missing_pct", 0.0),
            })
        missing_df = pd.DataFrame(rows)

        st.markdown("### Missing Values (Top 12)")
        st.dataframe(missing_df.sort_values("missing_pct", ascending=False).head(12))
    
    

    # Cardinality preview
    if "cardinality" in eda_summary:
        st.markdown("#### Categorical Cardinality (Preview)")
        top_card = list(eda_summary["cardinality"].items())[:12]
        st.write(dict(top_card))

    # Correlation
    corr = safe_path(eda_summary.get("corr_heatmap"))
    if corr:
        st.image(corr, caption="Correlation Heatmap", use_container_width=True)

    mm = safe_path(eda_summary.get("missing_matrix"))
    if mm:
        st.image(mm, caption="Missingness Matrix", use_container_width=True)

    # =====================================================
    # TARGET + AUTO-CLEAN
    # =====================================================
    st.subheader("🎯 Target & Auto-Clean Settings")
    target = st.selectbox("Select target column", df.columns)
    st.session_state.target = target   # persist target

    # -------- Target Cardinality Warning (Classification Only) --------
    task_type_guess = "Classification" if df[target].dtype == object else "Regression"

    # Only warn for classification tasks
    if task_type_guess != "Regression":
        unique_vals = df[target].nunique()
        if unique_vals > 50:
            st.warning(
                f"⚠ High target cardinality detected: **{unique_vals} unique classes**.\n"
                "This may cause poor performance. Consider grouping or encoding."
            )
    



    auto_clean = st.checkbox("Enable Auto-clean", value=True)
    drop_thresh = st.slider(
        "Drop columns with missing ratio above:",
        0.0,
        0.9,
        0.5,
        0.05,
        key="drop_thresh",
    )
    fill_numeric = st.selectbox(
        "Numeric fill strategy", ["mean", "median", "0"],
        index=0, key="fill_numeric"
    )
    fill_cat = st.selectbox(
        "Categorical fill strategy", ["most_frequent", "unknown"],
        index=0, key="fill_cat"
    )

    # Validate before cleaning
    # issues = eda.validate_data(df, target=target, high_missing_thresh=drop_thresh)
    # if issues:
    #     st.markdown("### ⚠️ Data Issues Detected")
    #     for it in issues:
    #         st.write("- " + it)
    # else:
    #     st.success("No immediate issues detected.")

    # =====================================================
    # RUN AUTO-CLEAN BUTTON
    # =====================================================
    if auto_clean and st.button("Run Auto-clean"):
        with st.spinner("Cleaning dataset..."):
            cleaned_df, report = eda.clean_data(
                df,
                target=target,
                drop_thresh=drop_thresh,
                fill_numeric=fill_numeric,
                fill_categorical=fill_cat,
            )

        st.session_state.cleaned_df = cleaned_df

        # Auto-clean summary
        st.subheader("🧹 Auto-Clean Summary")

        dropped_cols = report.get("dropped_columns", [])
        if dropped_cols:
            st.write(f"**Dropped columns:** {dropped_cols}")
        else:
            st.write("No columns dropped.")

        st.write(f"**Numeric columns imputed ({fill_numeric})**: {report.get('imputed_numeric', [])}")
        st.write(f"**Categorical columns imputed ({fill_cat})**: {report.get('imputed_categorical', [])}")
        st.write(f"**Rows dropped (missing target):** {report.get('dropped_rows_missing_target', 0)}")

        rows_after = report.get("rows_after", cleaned_df.shape[0])
        cols_after = report.get("columns_after", cleaned_df.shape[1])
        st.success(f"Cleaned dataset shape: **{rows_after} rows × {cols_after} columns**")

        st.markdown("#### Cleaned Data Preview")
        st.dataframe(cleaned_df.head())

    elif not auto_clean:
        st.session_state.cleaned_df = df
        st.info("Auto-clean disabled. Raw dataset will be used for training.")

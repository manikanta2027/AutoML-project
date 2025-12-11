# eda.py — robust EDA + clean_data + validate_data (with nicer plots)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Optional SMOTE flag
try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False


def run_eda(df, out_dir="reports/eda"):
    """
    Run lightweight EDA and save plots / summaries into out_dir.
    Returns a summary dict with paths and basic stats.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary = {}

    # --------- Basic info ----------
    summary["rows"] = int(df.shape[0])
    summary["columns"] = int(df.shape[1])
    summary["column_types"] = df.dtypes.apply(lambda x: str(x)).to_dict()

    # --------- Missing summary ----------
    missing = df.isna().sum()
    missing_pct = (missing / max(1, len(df))).round(4)
    miss_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    summary["missing"] = miss_df.to_dict()

    # --------- Duplicates ----------
    summary["duplicates"] = int(df.duplicated().sum())

    # --------- Categorical cardinality ----------
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    card = {c: int(df[c].nunique(dropna=True)) for c in cat_cols}
    summary["cardinality"] = card

    # --------- Numeric distributions & boxplots (top 12) ----------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols[:12]:
        col_data = df[c].dropna()
        if col_data.empty:
            continue

        # Histogram
        plt.figure(figsize=(5, 3), dpi=120)
        sns.histplot(col_data, kde=True)
        plt.title(f"Distribution: {c}")
        plt.xlabel(c)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dist_{c}.png"), bbox_inches="tight")
        plt.close()

        # Boxplot
        plt.figure(figsize=(5, 2.5), dpi=120)
        sns.boxplot(x=col_data)
        plt.title(f"Boxplot: {c}")
        plt.xlabel(c)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"box_{c}.png"), bbox_inches="tight")
        plt.close()

    # --------- Categorical bar plots (top 12 cols, top 10 cats each) ----------
    for c in cat_cols[:12]:
        vc = df[c].value_counts().nlargest(10)
        if vc.empty:
            continue

        plt.figure(figsize=(6, 3), dpi=120)
        sns.barplot(x=vc.values, y=vc.index)
        plt.title(f"Top categories: {c}")
        plt.xlabel("Count")
        plt.ylabel(c)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cat_{c}.png"), bbox_inches="tight")
        plt.close()

    # --------- Correlation heatmap (improved size + labels) ----------
   # --------- Improved Correlation Heatmap ----------
    if len(num_cols) >= 2:
        if len(num_cols) > 25:
            variances = df[num_cols].var().sort_values(ascending=False)
            top_cols = variances.head(25).index
            corr_df = df[top_cols]
        else:
            corr_df = df[num_cols]

        corr = corr_df.corr()

        plt.figure(figsize=(8, 6), dpi=140)

        sns.heatmap(
            corr,
            cmap="Blues",          # clean, modern, not ugly red/blue
            annot=False,
            square=False,
            linewidths=0.5,
            cbar_kws={"shrink": 0.6},
        )
    
        plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)

        corr_path = os.path.join(out_dir, "corr_heatmap.png")
        plt.tight_layout()
        plt.savefig(corr_path, bbox_inches="tight")
        plt.close()

        summary["corr_heatmap"] = corr_path


    # --------- Missingness matrix (better colors & labels) ----------
    # --------- Missingness Matrix (Improved visibility) ----------
    try:
        import missingno as msno

        fig = plt.figure(figsize=(10, 4), dpi=150)

        msno.matrix(
            df,
            fontsize=10,        # bigger labels
            sparkline=False,
            color=(0.2, 0.4, 0.8),   # modern blue tone
        )

        plt.title("Missingness Matrix", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)

        miss_path = os.path.join(out_dir, "missing_matrix.png")
        plt.tight_layout()
        plt.savefig(miss_path, bbox_inches="tight")
        plt.close()

        summary["missing_matrix"] = miss_path

    except Exception:
        pass

    # --------- describe() snapshot ----------
    try:
        desc = df.describe(include="all")
    except Exception:
        desc = df.describe()
    desc_path = os.path.join(out_dir, "describe.csv")
    desc.to_csv(desc_path)
    summary["describe_csv"] = desc_path

    return summary


def validate_data(df, target=None, max_cardinality=1000, high_missing_thresh=0.5):
    """
    Run basic validation checks and return a list of textual issues.
    """
    issues = []
    n, m = df.shape
    if n == 0 or m == 0:
        issues.append("Empty dataset (no rows or columns).")
        return issues

    # Target checks
    # if target is not None:
    #     if target not in df.columns:
    #         issues.append(f"Target '{target}' not in dataframe.")
    #     else:
    #         missing_target = df[target].isna().sum()
    #         if missing_target > 0:
    #             issues.append(
    #                 f"Target '{target}' has {missing_target} missing values (will be dropped)."
    #             )
    #         unique_target = df[target].nunique(dropna=True)
    #         issues.append(f"Target cardinality: {unique_target}")

    # High-missing columns
    miss = (df.isna().sum() / max(1, len(df))).sort_values(ascending=False)
    high_missing = miss[miss > high_missing_thresh]
    for col, pct in high_missing.items():
        issues.append(f"Column '{col}' has missing ratio {pct:.2%}")

    # High cardinality categoricals
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for c in cat_cols:
        unique = df[c].nunique(dropna=True)
        if unique > max_cardinality:
            issues.append(
                f"Categorical column '{c}' has high cardinality: {unique}"
            )

    # Duplicate rows
    dup = df.duplicated().sum()
    if dup > 0:
        issues.append(f"{dup} duplicate rows found")

    # Low-variance numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].nunique(dropna=True) <= 1:
            issues.append(f"Numeric column '{c}' has low variance (<=1 unique)")

    # Mixed types (rough check)
    for c in df.columns:
        non_na = df[c].dropna()
        if non_na.empty:
            continue
        sample = non_na.sample(min(100, len(non_na)))
        types = set(type(x) for x in sample)
        if any(t is str for t in types) and any(
            t in (int, float, np.int64, np.float64) for t in types
        ):
            issues.append(f"Column '{c}' has mixed types (strings + numbers)")

    return issues


def clean_data(df, target=None, drop_thresh=0.5,
               fill_numeric='mean', fill_categorical='most_frequent'):
    """
    Returns:
        cleaned_df
        report = {
            'dropped_columns': [...],
            'imputed_numeric': [...],
            'imputed_categorical': [...],
            'dropped_rows_missing_target': n,
            'rows_after': int,
            'columns_after': int
        }
    """
    df = df.copy()
    report = {}

    # 1. Drop high-missing columns
    miss_pct = df.isna().sum() / max(1, len(df))
    to_drop = miss_pct[miss_pct > drop_thresh].index.tolist()
    df = df.drop(columns=to_drop, errors="ignore")
    report['dropped_columns'] = to_drop

    # 2. Impute numeric
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report['imputed_numeric'] = num_cols.copy()

    for c in num_cols:
        if fill_numeric == 'mean':
            df[c] = df[c].fillna(df[c].mean())
        elif fill_numeric == 'median':
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(fill_numeric)

    # 3. Impute categorical
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    report['imputed_categorical'] = cat_cols.copy()

    for c in cat_cols:
        if fill_categorical == 'most_frequent':
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "")
        else:
            df[c] = df[c].fillna(fill_categorical)

    # 4. Drop rows with missing target
    if target is not None and target in df.columns:
        missing_target = df[target].isna().sum()
        df = df.dropna(subset=[target])
        report['dropped_rows_missing_target'] = int(missing_target)
    else:
        report['dropped_rows_missing_target'] = 0

    report['rows_after'] = df.shape[0]
    report['columns_after'] = df.shape[1]

    return df, report

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, rare_thresh=0.03, skew_thresh=0.75):
        self.rare_thresh = rare_thresh
        self.skew_thresh = skew_thresh
        self.rare_maps = {}
        self.skew_cols = []
        self.numeric_cols = []
        self.categorical_cols = []

    def fit(self, X, y=None):
        df = X.copy()

        # Detect column types
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 1) Rare categories
        for col in self.categorical_cols:
            freq = df[col].value_counts(normalize=True)
            rare_vals = freq[freq < self.rare_thresh].index
            self.rare_maps[col] = list(rare_vals)

        # 2) Skewed columns (log-transform safe)
        for col in self.numeric_cols:
            try:
                if abs(df[col].skew()) > self.skew_thresh:
                    self.skew_cols.append(col)
            except:
                pass

        return self

    def transform(self, X):
        df = X.copy()

        # 1) Replace rare categories
        for col, rare_vals in self.rare_maps.items():
            df[col] = df[col].replace(rare_vals, "RARE")

        # 2) Log transform skewed columns
        for col in self.skew_cols:
            try:
                df[col] = np.log1p(df[col].abs())
            except:
                pass

        return df


# -------------------------------------------------------
# Wrapper required for app.py
# -------------------------------------------------------

def auto_feature_engineering(df):
    """Wrapper for compatibility with app.py.
    It returns the SAME df (no new columns), plus a small summary.
    """

    fe = AdvancedFeatureEngineer()
    fe.fit(df)
    df_out = fe.transform(df)

    summary = {
        "numeric_string_converted": [],
        "polynomial_features_added": [],
        "outliers_removed": 0,
        "rare_categories_fixed": fe.rare_maps,
        "skewed_columns_transformed": fe.skew_cols,
    }

    return df_out, summary

# classification_engine.py

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except:
    SMOTE = None
    _HAS_SMOTE = False


def train_classification_models(df, target, seed=42, test_size=0.2, apply_smote=False):
    """
    Trains multiple classification models with preprocessing + polynomial features.
    Returns:
    - cv_scores
    - test_scores
    - trained_models
    - X_train, X_test, y_train, y_test
    """

    # -----------------------
    # Split
    # -----------------------
    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    # -----------------------
    # Preprocess
    # -----------------------
    numeric = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler())
        ]), numeric),
        
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    # -----------------------
    # Models
    # -----------------------
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "SVC": SVC(probability=True),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "HistGradientBoosting": HistGradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "KNN": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
    }

    cv_scores = {}
    test_scores = {}
    trained_models = {}

    # -----------------------
    # Optional SMOTE
    # -----------------------
    if apply_smote and _HAS_SMOTE:
        try:
            sm = SMOTE(random_state=seed)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        except Exception:
            X_train_res, y_train_res = X_train, y_train
    else:
        X_train_res, y_train_res = X_train, y_train

    # -----------------------
    # Train loop
    # -----------------------
    for name, model in models.items():

        pipe = Pipeline([
            ("preprocess", preprocess),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", model)
        ])

        # Cross-validation
        try:
            cv = cross_val_score(pipe, X_train_res, y_train_res, cv=5, scoring="accuracy")
            cv_scores[name] = float(np.mean(cv))
        except Exception as e:
            cv_scores[name] = f"FAILED: {str(e)[:80]}"

        # Fit & test
        try:
            pipe.fit(X_train_res, y_train_res)
            trained_models[name] = pipe

            test_acc = pipe.score(X_test, y_test)
            test_scores[name] = float(test_acc)

        except Exception as e:
            test_scores[name] = f"FAILED: {str(e)[:80]}"

    # -----------------------
    # Return everything
    # -----------------------
    return (
        cv_scores,
        test_scores,
        trained_models,
        X_train, X_test, y_train, y_test
    )

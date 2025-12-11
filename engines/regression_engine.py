# regression_engine.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


def train_regression_models(df, target, seed=42, test_size=0.2):

    # -----------------------
    # SPLIT
    # -----------------------
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # -----------------------
    # PREPROCESSING
    # -----------------------
    numeric = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

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
    # MODEL CATEGORIES
    # -----------------------
    linear_models = {
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
    }

    nonlinear_models = {
        "SVR": SVR(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "KNN": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(eval_metric="rmse"),
    }

    cv_scores = {}
    test_scores = {}
    trained_models = {}

    # -----------------------
    # TRAIN LINEAR MODELS (with polynomial features)
    # -----------------------
    for name, model in linear_models.items():

        pipe = Pipeline([
            ("preprocess", preprocess),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", model)
        ])

        # CV
        try:
            cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
            cv_scores[name] = float(np.mean(cv))
        except Exception as e:
            cv_scores[name] = f"FAILED: {str(e)[:80]}"

        # Fit & Test
        try:
            pipe.fit(X_train, y_train)
            test_scores[name] = float(pipe.score(X_test, y_test))
            trained_models[name] = pipe
        except Exception as e:
            test_scores[name] = f"FAILED: {str(e)[:80]}"

    # -----------------------
    # TRAIN NON-LINEAR MODELS (NO poly features)
    # -----------------------
    for name, model in nonlinear_models.items():

        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", model)
        ])

        # CV
        try:
            cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
            cv_scores[name] = float(np.mean(cv))
        except Exception as e:
            cv_scores[name] = f"FAILED: {str(e)[:80]}"

        # Fit & Test
        try:
            pipe.fit(X_train, y_train)
            test_scores[name] = float(pipe.score(X_test, y_test))
            trained_models[name] = pipe
        except Exception as e:
            test_scores[name] = f"FAILED: {str(e)[:80]}"

    # -----------------------
    # RETURN
    # -----------------------
    return (
        cv_scores,
        test_scores,
        trained_models,
        X_train, X_test, y_train, y_test
    )

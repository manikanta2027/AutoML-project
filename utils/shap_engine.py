# utils/shap_engine.py
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap  # type: ignore
except ImportError:  # pragma: no cover
    shap = None


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _safe_sample_df(
    X: pd.DataFrame,
    sample_frac: float,
    max_samples: int,
    random_state: int
) -> pd.DataFrame:
    """Return a sampled DataFrame (never empty, max=max_samples)."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    n = len(X)
    if n == 0:
        raise ValueError("X has 0 rows; cannot compute SHAP.")

    target_n = max(1, int(n * sample_frac))
    target_n = min(target_n, max_samples, n)
    return X.sample(n=target_n, random_state=random_state)


def _is_tree_model(est) -> bool:
    """Heuristic: is this a tree-based model? (RF, GBM, XGB, etc.)."""
    if est is None:
        return False
    name = est.__class__.__name__.lower()
    tree_keywords = ["forest", "tree", "xgb", "gbm", "boost", "histgradient"]
    if any(k in name for k in tree_keywords):
        return True
    if hasattr(est, "tree_") or hasattr(est, "estimators_"):
        return True
    return False


def _get_estimator_from_pipeline(model):
    """If model is a Pipeline, return inner estimator; else return model."""
    est = model
    if hasattr(model, "steps"):
        est = model.steps[-1][1]
    return est


def compute_shap_and_save(
    model,
    X: pd.DataFrame,
    sample_frac: float = 0.10,
    max_samples: int = 400,
    random_state: int = 42,
    top_k_dependence: int = 6,
    out_dir: str = "reports/shap",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Lightweight SHAP: only supports TREE models via TreeExplainer.

    Returns
    -------
    saved : dict of image paths
    ctx   : dict with debug info / errors
    """
    ctx: Dict[str, Any] = {
        "method": None,
        "error": None,
        "n_rows_X": len(X) if X is not None else 0,
    }
    saved: Dict[str, Any] = {
        "summary": None,
        "bar": None,
        "dependence": [],
        "waterfall": None,
        "force_png": None,
        "force_html": None,
    }

    if shap is None:
        ctx["error"] = "shap is not installed."
        return saved, ctx

    if X is None or len(X) == 0:
        ctx["error"] = "X is empty; cannot compute SHAP."
        return saved, ctx

    # basic size guard (extra safety)
    n_rows, n_features = X.shape
    if n_rows > 3000 or n_features > 120:
        ctx["error"] = f"Dataset too large for SHAP (rows={n_rows}, features={n_features})."
        return saved, ctx

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    out_dir = _ensure_dir(out_dir)
    base_name = getattr(model, "__class__", type("X", (), {})).__name__

    # -------------------- only tree models --------------------
    est = _get_estimator_from_pipeline(model)
    if not _is_tree_model(est):
        ctx["error"] = "Non-tree model: SHAP disabled in this version."
        return saved, ctx

    ctx["method"] = "TreeExplainer"

    try:
        # sample rows
        X_sample = _safe_sample_df(X, sample_frac, max_samples, random_state)
        ctx["n_shap_rows"] = len(X_sample)

        # If model is a pipeline, we feed the transformed data into TreeExplainer
        if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
            X_trans = model.named_steps["preprocess"].transform(X_sample)
        else:
            X_trans = X_sample

        explainer = shap.TreeExplainer(est)
        shap_values = explainer.shap_values(X_trans)

        # For multi-class tree models, pick the last class
        if isinstance(shap_values, list):
            shap_arr = np.array(shap_values[-1])
        else:
            shap_arr = np.array(shap_values)

        if shap_arr.ndim == 3:
            shap_arr = shap_arr[-1]

        # ---------------- summary plot ----------------
        plt.figure()
        shap.summary_plot(shap_arr, X_trans, show=False)
        summary_path = os.path.join(out_dir, f"{base_name}_shap_summary.png")
        plt.tight_layout()
        plt.gcf().set_size_inches(6.0, 3.2)
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        if os.path.exists(summary_path):
            saved["summary"] = summary_path

        # ---------------- bar plot ----------------
        plt.figure()
        shap.summary_plot(shap_arr, X_trans, plot_type="bar", show=False)
        bar_path = os.path.join(out_dir, f"{base_name}_shap_bar.png")
        plt.tight_layout()
        plt.gcf().set_size_inches(5.5, 3.6)
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        if os.path.exists(bar_path):
            saved["bar"] = bar_path

        # ---------------- dependence plots (top-k) ----------------
        mean_abs = np.abs(shap_arr).mean(axis=0)
        idx_sorted = np.argsort(-mean_abs)
        top_idx = idx_sorted[:top_k_dependence]

        dep_paths: List[str] = []
        for i, f_idx in enumerate(top_idx):
            try:
                plt.figure()
                shap.dependence_plot(f_idx, shap_arr, X_trans, show=False)
                dep_path = os.path.join(out_dir, f"{base_name}_shap_dep_{i+1}.png")
                plt.tight_layout()
                plt.gcf().set_size_inches(5.5, 3.8)
                plt.savefig(dep_path, dpi=150, bbox_inches="tight")
                plt.close()
                if os.path.exists(dep_path):
                    dep_paths.append(dep_path)
            except Exception:
                plt.close("all")
                continue

        saved["dependence"] = dep_paths

        return saved, ctx

    except Exception as e:
        ctx["error"] = str(e)
        return saved, ctx

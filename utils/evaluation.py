# evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os

sns.set_style("whitegrid")

def plot_confusion_matrix(model, X, y, save_path):
    try:
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path); plt.close()
    except Exception:
        return

def plot_roc_curve(model, X, y, save_path):
    try:
        y_score = model.predict_proba(X)
        # if multiclass, compute macro-average for second class may not exist; handle binary
        if y_score.shape[1] == 2:
            y_prob = y_score[:,1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(5,4))
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            plt.plot([0,1],[0,1],"k--")
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.tight_layout(); plt.savefig(save_path); plt.close()
    except Exception:
        return

def plot_precision_recall_curve(model, X, y, save_path):
    try:
        y_score = model.predict_proba(X)
        if y_score.shape[1] == 2:
            y_prob = y_score[:,1]
            p, r, _ = precision_recall_curve(y, y_prob)
            plt.figure(figsize=(5,4))
            plt.plot(r, p)
            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.tight_layout(); plt.savefig(save_path); plt.close()
    except Exception:
        return

def plot_feature_importance(model, feature_names, save_path):
    try:
        # if pipeline, get final estimator
        final = model
        if hasattr(model, "named_steps"):
            final = model.named_steps.get("model", model)
        if hasattr(final, "feature_importances_"):
            importances = final.feature_importances_
            idx = np.argsort(importances)
            plt.figure(figsize=(6,5))
            plt.barh(np.array(feature_names)[idx], importances[idx])
            plt.title("Feature importance")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.tight_layout(); plt.savefig(save_path); plt.close()
        elif hasattr(final, "coef_"):
            coef = np.ravel(final.coef_)
            if coef.shape[0] == len(feature_names):
                idx = np.argsort(np.abs(coef))
                plt.figure(figsize=(6,5))
                plt.barh(np.array(feature_names)[idx], coef[idx])
                plt.title("Feature coefficients")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.tight_layout(); plt.savefig(save_path); plt.close()
    except Exception:
        return

def plot_pred_vs_actual(y_true, y_pred, save_path):
    try:
        plt.figure(figsize=(5,4))
        plt.scatter(y_true, y_pred, alpha=0.6)
        mn = min(min(y_true), min(y_pred))
        mx = max(max(y_true), max(y_pred))
        plt.plot([mn,mx],[mn,mx],"r--")
        plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Predicted vs Actual")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout(); plt.savefig(save_path); plt.close()
    except Exception:
        return

def plot_residuals(y_true, y_pred, save_path):
    try:
        residuals = np.array(y_true) - np.array(y_pred)
        plt.figure(figsize=(5,4))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color="r", linestyle="--")
        plt.xlabel("Predicted"); plt.ylabel("Residuals"); plt.title("Residuals")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout(); plt.savefig(save_path); plt.close()
    except Exception:
        return

def plot_corr_heatmap(df, path):
    """Generate a compact, high-contrast correlation heatmap."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))  # Smaller size
    sns.set_style("whitegrid")

    ax = sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        linewidths=0.3,
        linecolor="gray",
        cbar=True
    )

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


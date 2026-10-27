"""
Binary Classification Module
============================
Trains and evaluates models for overall Machine Failure prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc,
)
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import joblib


def get_models():
    """Return dictionary of classification models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=10, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
    }


def evaluate_models(X_train, y_train, X_test, y_test, output_dir="reports/figures"):
    """
    Train, cross-validate, and evaluate all models.
    Returns results DataFrame and best model info.
    """
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_f1 = 0
    best_info = None

    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

        # Train on full training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Get probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            model_calib = CalibratedClassifierCV(model, cv=3)
            model_calib.fit(X_train, y_train)
            y_proba = model_calib.predict_proba(X_test)[:, 1]

        # Metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        p_curve, r_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(r_curve, p_curve)

        results.append({
            "Model": name,
            "CV F1 (mean)": f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}",
            "Accuracy": f"{acc:.4f}",
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "F1-Score": f"{f1:.4f}",
            "ROC-AUC": f"{roc:.4f}",
            "PR-AUC": f"{pr_auc:.4f}",
        })

        print(f"  [{name:22s}] F1={f1:.4f} | ROC-AUC={roc:.4f} | PR-AUC={pr_auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_info = (name, model, y_pred, y_proba)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/../binary_classification_results.csv", index=False)

    # Plot ROC and PR curves
    _plot_curves(models, X_train, y_train, X_test, y_test, output_dir)

    return results_df, best_info


def _plot_curves(models, X_train, y_train, X_test, y_test, output_dir):
    """Plot ROC and PR curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, curve_type in zip(axes, ["ROC", "PR"]):
        for name, model in models.items():
            # Get probabilities
            if hasattr(model, "predict_proba"):
                model.fit(X_train, y_train)
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                model_calib = CalibratedClassifierCV(model, cv=3)
                model_calib.fit(X_train, y_train)
                y_proba = model_calib.predict_proba(X_test)[:, 1]

            if curve_type == "ROC":
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                score = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={score:.3f})")
            else:
                p, r, _ = precision_recall_curve(y_test, y_proba)
                score = auc(r, p)
                ax.plot(r, p, lw=2, label=f"{name} (AUC={score:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6) if curve_type == "ROC" else             ax.axhline(y=y_test.sum() / len(y_test), color="gray", linestyle="--", alpha=0.6,
                       label=f"Baseline ({y_test.mean():.3f})")
        ax.set_xlabel("False Positive Rate" if curve_type == "ROC" else "Recall")
        ax.set_ylabel("True Positive Rate" if curve_type == "ROC" else "Precision")
        ax.set_title(f"{'ROC' if curve_type == 'ROC' else 'Precision-Recall'} Curves",
                     fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/07_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] ROC/PR curves saved")


def plot_feature_importance(model, feature_names, output_dir="reports/figures"):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print("[!] Model does not support feature importances")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    bars = ax.barh(range(len(indices)), importances[indices], color="#3498db", edgecolor="black")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Feature Importance - Machine Failure Prediction", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/08_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Feature importance plot saved")


def threshold_analysis(y_test, y_proba, output_dir="reports/figures"):
    """Find optimal decision threshold that maximizes F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thresholds, precisions[:-1], "b-", lw=2, label="Precision")
    ax.plot(thresholds, recalls[:-1], "r-", lw=2, label="Recall")
    ax.plot(thresholds, f1_scores, "g-", lw=2, label="F1-Score")
    ax.axvline(x=best_threshold, color="gray", linestyle="--", alpha=0.7,
               label=f"Best Threshold = {best_threshold:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/09_threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Threshold analysis saved (best F1 threshold: {best_threshold:.4f})")
    return best_threshold


def save_model(model, name, output_dir="models"):
    """Save trained model to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = f"{output_dir}/{name.lower().replace(' ', '_')}.joblib"
    joblib.dump(model, path)
    print(f"[OK] Model saved: {path}")
    return path


if __name__ == "__main__":
    print("Run this module from the main pipeline or notebook.")
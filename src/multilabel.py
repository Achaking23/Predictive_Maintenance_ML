"""
Multi-Label Classification Module
==================================
Trains and evaluates models for individual failure type prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, hamming_loss,
    precision_recall_curve, auc, label_ranking_average_precision_score,
)
from pathlib import Path


FAILURE_LABELS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_DESCRIPTIONS = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}


def train_multilabel(X_train, y_train, X_test, y_test, output_dir="reports/figures"):
    """
    Train one binary classifier per failure type (Binary Relevance).
    Uses weighted training to handle class imbalance.

    Returns:
        models_per_type (dict): Per-failure-type model info
    """
    models_per_type = {}

    print("\n" + "=" * 60)
    print("Multi-Label Classification - Training Per Failure Type")
    print("=" * 60)

    for ft in FAILURE_LABELS:
        y_ft_train = y_train[ft].values
        y_ft_test = y_test[ft].values

        # Compute class weights
        neg_count = (y_ft_train == 0).sum()
        pos_count = (y_ft_train == 1).sum()
        weight_ratio = neg_count / max(pos_count, 1)
        sample_weights = np.where(y_ft_train == 1, weight_ratio, 1.0)

        # Train
        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
        )
        model.fit(X_train, y_ft_train, sample_weight=sample_weights)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Evaluate
        prec = precision_score(y_ft_test, y_pred, zero_division=0)
        rec = recall_score(y_ft_test, y_pred, zero_division=0)
        f1 = f1_score(y_ft_test, y_pred, zero_division=0)
        p_curve, r_curve, _ = precision_recall_curve(y_ft_test, y_proba)
        pr_auc = auc(r_curve, p_curve)

        models_per_type[ft] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "pr_auc": pr_auc,
            "support": y_ft_test.sum(),
        }

        print(f"  [{ft:6s}] {FAILURE_DESCRIPTIONS[ft]:30s} | "
              f"Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} PR-AUC={pr_auc:.3f}")

    return models_per_type


def evaluate_multilabel(models_per_type, y_test, output_dir="reports/figures"):
    """
    Compute overall multi-label metrics and generate visualizations.
    """
    y_pred_all = np.column_stack([models_per_type[ft]["y_pred"] for ft in FAILURE_LABELS])
    y_proba_all = np.column_stack([models_per_type[ft]["y_proba"] for ft in FAILURE_LABELS])
    y_test_arr = y_test.values

    # Overall metrics
    hl = hamming_loss(y_test_arr, y_pred_all)
    exact_match = (y_pred_all == y_test_arr).all(axis=1).mean()
    micro_f1 = f1_score(y_test_arr, y_pred_all, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test_arr, y_pred_all, average="macro", zero_division=0)

    try:
        lrap = label_ranking_average_precision_score(y_test_arr, y_proba_all)
    except Exception:
        lrap = None

    print("\n" + "=" * 60)
    print("OVERALL MULTI-LABEL METRICS")
    print("=" * 60)
    print(f"  Hamming Loss:        {hl:.6f}  (lower is better)")
    print(f"  Exact Match Acc:     {exact_match:.4f}")
    print(f"  Micro F1:            {micro_f1:.4f}")
    print(f"  Macro F1:            {macro_f1:.4f}")
    if lrap:
        print(f"  Label Ranking AP:    {lrap:.4f}")

    # Plot per-type metrics
    _plot_multilabel_metrics(models_per_type, output_dir)
    _plot_confusion_matrices(models_per_type, y_test, output_dir)

    # Save results
    results = []
    for ft in FAILURE_LABELS:
        m = models_per_type[ft]
        results.append({
            "Failure Type": ft,
            "Description": FAILURE_DESCRIPTIONS[ft],
            "Test Positives": m["support"],
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1": f"{m['f1']:.4f}",
            "PR-AUC": f"{m['pr_auc']:.4f}",
        })
    pd.DataFrame(results).to_csv(f"{output_dir}/../multilabel_results.csv", index=False)

    return {"hamming_loss": hl, "exact_match": exact_match, "micro_f1": micro_f1, "macro_f1": macro_f1}


def _plot_multilabel_metrics(models_per_type, output_dir):
    """Bar chart comparing metrics across failure types."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(FAILURE_LABELS))
    width = 0.2
    metrics_data = {
        "Precision": [models_per_type[ft]["precision"] for ft in FAILURE_LABELS],
        "Recall": [models_per_type[ft]["recall"] for ft in FAILURE_LABELS],
        "F1-Score": [models_per_type[ft]["f1"] for ft in FAILURE_LABELS],
        "PR-AUC": [models_per_type[ft]["pr_auc"] for ft in FAILURE_LABELS],
    }
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    for i, (metric, values) in enumerate(metrics_data.items()):
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i], edgecolor="black")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(FAILURE_LABELS)
    ax.set_ylabel("Score")
    ax.set_title("Performance Metrics per Failure Type", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/11_multilabel_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Multi-label metrics plot saved")


def _plot_confusion_matrices(models_per_type, y_test, output_dir):
    """Plot confusion matrices for each failure type."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, ft in enumerate(FAILURE_LABELS):
        ax = axes[i]
        m = models_per_type[ft]
        y_true = y_test[ft].values.astype(bool)
        y_pred = m["y_pred"].astype(bool)

        cm = np.zeros((2, 2))
        cm[0, 0] = (~y_true & ~y_pred).sum()
        cm[0, 1] = (~y_true & y_pred).sum()
        cm[1, 0] = (y_true & ~y_pred).sum()
        cm[1, 1] = (y_true & y_pred).sum()

        ax.matshow(cm, cmap="Blues", alpha=0.8)
        for (row, col), val in np.ndenumerate(cm):
            ax.text(col, row, f"{int(val)}", ha="center", va="center", fontsize=12, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{ft} - {FAILURE_DESCRIPTIONS[ft]}\nF1={m['f1']:.3f}", fontweight="bold")

    axes[5].axis("off")
    plt.suptitle("Confusion Matrices per Failure Type", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/10_multilabel_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Multi-label confusion matrices saved")


if __name__ == "__main__":
    print("Run this module from the main pipeline or notebook.")
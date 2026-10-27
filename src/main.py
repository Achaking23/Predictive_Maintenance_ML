"""
Predictive Maintenance ML Pipeline
===================================
Entry point for the complete ML pipeline.
Usage: python -m src.main
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.eda import (
    load_data,
    plot_target_distributions,
    plot_correlation_matrix,
    plot_feature_distributions,
)
from src.preprocessing import engineer_features, prepare_targets, split_and_scale
from src.models import evaluate_models, plot_feature_importance, threshold_analysis, save_model
from src.multilabel import train_multilabel, evaluate_multilabel


def main():
    print("=" * 80)
    print("  PREDICTIVE MAINTENANCE ML PIPELINE")
    print("=" * 80)

    # Configuration
    DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
    OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
    MODELS_DIR = PROJECT_ROOT / "models"

    # 1. Load Data
    print("\n[1/6] Loading data...")
    df = load_data(str(DATA_PATH))

    # 2. EDA
    print("\n[2/6] Exploratory Data Analysis...")
    plot_target_distributions(df, str(OUTPUT_DIR))
    plot_correlation_matrix(df, str(OUTPUT_DIR))
    plot_feature_distributions(df, str(OUTPUT_DIR))

    # 3. Feature Engineering
    print("\n[3/6] Feature Engineering...")
    X, feature_names = engineer_features(df)
    y_binary, y_multilabel = prepare_targets(df)

    # 4. Split & Scale
    print("\n[4/6] Train/Test Split & Scaling...")
    data = split_and_scale(X, y_binary, y_multilabel)

    # 5. Binary Classification
    print("\n[5/6] Binary Classification...")
    results_df, best_info = evaluate_models(
        data["X_train"], data["y_train"],
        data["X_test"], data["y_test"],
        str(OUTPUT_DIR),
    )

    # Feature importance
    rf = __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(data["X_train"], data["y_train"])
    plot_feature_importance(rf, data["feature_names"], str(OUTPUT_DIR))

    # Threshold analysis
    best_threshold = threshold_analysis(data["y_test"], best_info[3], str(OUTPUT_DIR))

    # Save best model
    save_model(best_info[1], best_info[0], str(MODELS_DIR))

    # 6. Multi-Label Classification
    print("\n[6/6] Multi-Label Classification...")
    models_per_type = train_multilabel(
        data["X_train"], data["y_train_ml"],
        data["X_test"], data["y_test_ml"],
        str(OUTPUT_DIR),
    )
    overall_metrics = evaluate_multilabel(models_per_type, data["y_test_ml"], str(OUTPUT_DIR))

    # Summary
    print("\n" + "=" * 80)
    print("  PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"  Best binary model: {best_info[0]} (F1={best_info[1].__class__.__name__})")
    print(f"  Optimal threshold: {best_threshold:.4f}")
    print(f"  Multi-label Hamming Loss: {overall_metrics['hamming_loss']:.6f}")
    print(f"  All outputs in: {OUTPUT_DIR.parent}")
    print("=" * 80)


if __name__ == "__main__":
    main()
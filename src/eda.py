"""
Exploratory Data Analysis (EDA) Module
=======================================
Generates visualizations and statistics to understand the dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """Load the AI4I 2020 predictive maintenance dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing values: {df.isnull().sum().sum()}")
    return df


def plot_target_distributions(df: pd.DataFrame, output_dir: str = "reports/figures"):
    """Plot distribution of each failure type."""
    failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, col in enumerate(failure_cols):
        ax = axes[i]
        counts = df[col].value_counts()
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax.bar(["No Failure", "Failure"], counts.values, color=colors, edgecolor="black")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                    f"{val}\n({val/len(df)*100:.2f}%)", ha="center", va="bottom", fontweight="bold")
        ax.set_title(f"{col} Distribution", fontweight="bold")

    axes[5].axis("off")
    plt.suptitle("Failure Type Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/01_target_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Target distributions plotted")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str = "reports/figures"):
    """Plot correlation heatmap of numeric features and targets."""
    numeric_features = ["Air temperature [K]", "Process temperature [K]",
                        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"]

    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df[numeric_features + failure_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.8, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Correlation matrix plotted")


def plot_feature_distributions(df: pd.DataFrame, output_dir: str = "reports/figures"):
    """Plot distribution of each numeric feature."""
    numeric_features = ["Air temperature [K]", "Process temperature [K]",
                        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, feat in enumerate(numeric_features):
        ax = axes[i]
        sns.histplot(df[feat], kde=True, bins=40, color="#3498db", alpha=0.7, ax=ax)
        ax.axvline(df[feat].mean(), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {df[feat].mean():.1f}")
        ax.axvline(df[feat].median(), color="green", linestyle="--", linewidth=2,
                   label=f"Median: {df[feat].median():.1f}")
        ax.set_title(f"{feat}", fontweight="bold")
        ax.legend(fontsize=8)

    axes[5].axis("off")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Feature distributions plotted")


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for the dataset."""
    return df.describe()


if __name__ == "__main__":
    print("Run this module from the main pipeline or notebook.")
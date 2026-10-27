"""
Utility Functions Module
========================
Helper functions for logging, configuration, and model persistence.
"""

import json
import yaml
from pathlib import Path
import joblib


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_metrics(metrics: dict, filepath: str):
    """Save evaluation metrics to JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved: {filepath}")


def load_model(filepath: str):
    """Load a trained model from disk."""
    return joblib.load(filepath)


def format_classification_report(report_dict: dict) -> str:
    """Format sklearn classification report as a readable string."""
    lines = []
    lines.append(f"{'':>20} {'precision':>10} {'recall':>8} {'f1-score':>10} {'support':>8}")
    lines.append("-" * 60)
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            lines.append(
                f"{label:>20} {metrics['precision']:>10.3f} {metrics['recall']:>8.3f} "
                f"{metrics['f1-score']:>10.3f} {metrics['support']:>8.0f}"
            )
    return "\n".join(lines)


def print_section_header(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


if __name__ == "__main__":
    print("Utility functions loaded successfully.")
"""
Preprocessing & Feature Engineering Module
===========================================
Handles encoding, scaling, and domain-driven feature creation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Create engineered features from raw sensor data.

    Returns:
        X (pd.DataFrame): Feature matrix with engineered features
        feature_names (list): List of feature column names
    """
    df_ml = df.copy()

    # Encode product type
    le = LabelEncoder()
    df_ml["Type_encoded"] = le.fit_transform(df_ml["Type"])  # H=0, L=1, M=2

    # Temperature-related
    df_ml["temperature_diff"] = df_ml["Process temperature [K]"] - df_ml["Air temperature [K]"]

    # Mechanical power proxy
    df_ml["power_like"] = df_ml["Torque [Nm]"] * df_ml["Rotational speed [rpm]"]

    # Torque-to-speed ratio (indicates operating condition)
    df_ml["torque_per_speed"] = df_ml["Torque [Nm]"] / (df_ml["Rotational speed [rpm]"] + 1)

    # Tool wear stage (binned)
    df_ml["wear_stage"] = pd.cut(
        df_ml["Tool wear [min]"],
        bins=[-1, 50, 100, 150, 200, 300],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    # Define feature columns
    feature_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type_encoded",
        "temperature_diff",
        "power_like",
        "torque_per_speed",
        "wear_stage",
    ]

    X = df_ml[feature_cols]
    print(f"Engineered {len(feature_cols)} features from {len(df.columns)} raw columns")
    return X, feature_cols


def prepare_targets(df: pd.DataFrame) -> tuple:
    """
    Extract binary and multi-label targets.

    Returns:
        y_binary (pd.Series): Machine failure (overall)
        y_multilabel (pd.DataFrame): Individual failure types
    """
    y_binary = df["Machine failure"]
    y_multilabel = df[["TWF", "HDF", "PWF", "OSF", "RNF"]]
    print(f"Binary target: Machine failure (rate: {y_binary.mean()*100:.2f}%)")
    print(f"Multi-label targets: 5 failure types")
    return y_binary, y_multilabel


def split_and_scale(X, y_binary, y_multilabel, test_size=0.2, random_state=42):
    """
    Perform train/test split and feature scaling.

    Returns:
        dict with X_train, X_test, y_train, y_test, scaler
    """
    # Stratify on binary target for consistency
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
    )

    # Split multi-label using same indices
    _, _, y_train_ml, y_test_ml = train_test_split(
        X, y_multilabel, test_size=test_size, random_state=random_state,
        stratify=(y_multilabel.sum(axis=1) > 0).astype(int),
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"Train failure rate: {y_train.mean()*100:.2f}% | Test: {y_test.mean()*100:.2f}%")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_ml": y_train_ml,
        "y_test_ml": y_test_ml,
        "scaler": scaler,
        "feature_names": X.columns.tolist(),
    }


if __name__ == "__main__":
    print("Run this module from the main pipeline or notebook.")
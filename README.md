# 🔧 Predictive Maintenance with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-AI4I%202020-purple)](https://www.kaggle.com/datasets)

> **A complete end-to-end Machine Learning pipeline for industrial predictive maintenance, 
> achieving 96.9% ROC-AUC and 85.9% F1-score on imbalanced machinery failure data.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Results Summary](#results-summary)
- [Key Insights](#key-insights)
- [Production Deployment](#production-deployment)
- [License](#license)

---

## 📖 Overview

This project applies **supervised machine learning** to predict industrial machinery failures 
from sensor readings. The goal is **predictive maintenance** — catching failures before they 
happen, reducing unplanned downtime, and optimizing maintenance schedules.

### What makes this project different?

| Aspect | Our Approach |
|--------|-------------|
| **Target** | Multi-label: overall failure + 5 individual failure types |
| **Imbalance** | Only 3.39% failure rate — evaluated with PR-AUC, not accuracy |
| **Models** | 5 classifiers compared + stacking ensemble |
| **Metrics** | F1-score, PR-AUC, ROC-AUC, Hamming Loss — chosen for business relevance |
| **Engineering** | Domain-driven features (torque/speed ratio, temperature delta, power proxy) |

---

## 📊 Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets)

The dataset contains **10,000 sensor readings** from industrial machinery:

| Feature | Description |
|---------|-------------|
| Air temperature [K] | Ambient air temperature (295.3–304.5 K) |
| Process temperature [K] | Process temperature (305.7–313.8 K) |
| Rotational speed [rpm] | Shaft speed (1,168–2,886 rpm) |
| Torque [Nm] | Applied torque (3.8–76.6 Nm) |
| Tool wear [min] | Tool usage time (0–253 min) |
| Product Type | H (1,003), L (6,000), M (2,997) |

**Target variables (binary):**

| Column | Failure Type | Rate |
|--------|-------------|------|
| `Machine failure` | Overall failure (any type) | **3.39%** |
| `HDF` | Heat Dissipation Failure | 1.15% |
| `PWF` | Power Failure | 0.95% |
| `OSF` | Overstrain Failure | 0.98% |
| `TWF` | Tool Wear Failure | 0.46% |
| `RNF` | Random Failure | 0.19% |

---

## 🏗️ Project Structure

```
Predictive_Maintenance_ML/
│
├── data/
│   └── ai4i2020.csv                          # Original dataset
│
├── notebooks/
│   └── predictive_maintenance_ml.ipynb        # Complete analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── eda.py                                # Exploratory Data Analysis
│   ├── preprocessing.py                       # Feature engineering & scaling
│   ├── models.py                              # Model training & evaluation
│   ├── multilabel.py                          # Multi-label classification
│   └── utils.py                               # Helper functions
│
├── models/
│   └── (trained models saved here)
│
├── reports/
│   ├── ML_Project_Report.html                 # Interactive HTML report
│   ├── figures/                               # All visualizations
│   │   ├── 01_target_distributions.png
│   │   ├── 02_product_type_analysis.png
│   │   ├── ... (11 figures total)
│   │   └── 11_multilabel_metrics.png
│   ├── binary_classification_results.csv
│   ├── multilabel_results.csv
│   └── consolidated_results.csv
│
├── config/
│   ├── model_config.yaml                      # Model hyperparameters
│   └── features.yaml                          # Feature definitions
│
├── requirements.txt                           # Python dependencies
├── setup.py                                   # Package installer
├── .gitignore
├── LICENSE                                     # MIT License
└── README.md                                  # ← You are here
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/Predictive_Maintenance_ML.git
cd Predictive_Maintenance_ML
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Option A: Run as a script
python -m src.main

# Option B: Open the notebook
jupyter notebook notebooks/predictive_maintenance_ml.ipynb
```

### 3. View Results

Open `reports/ML_Project_Report.html` in any browser for the interactive report.

---

## 🧪 Methodology

### Feature Engineering

Engineered features derived from domain knowledge:

- **`torque_per_speed`** — Torque / Rotational speed (anomalous ratios indicate failure)
- **`power_like`** — Torque × Rotational speed (proxy for mechanical power)
- **`temperature_diff`** — Process temp − Air temp (heat buildup indicator)
- **`wear_stage`** — Binned tool wear into 5 stages

### Models Compared

| Model | Why It's Here |
|-------|--------------|
| **Gradient Boosting** ★ | Best for imbalanced tabular data; handles non-linearity well |
| **Random Forest** | Robust baseline; provides feature importances |
| **Logistic Regression** | Linear baseline; interpretable coefficients |
| **Decision Tree** | Single-tree baseline; shows overfitting risk |
| **KNN** | Non-parametric baseline; sensitive to local patterns |

### Evaluation Metrics

| Metric | Why It Matters |
|--------|---------------|
| **F1-Score** | **Primary** — Harmonic mean of Precision & Recall. The cost of a false alarm (inspecting a healthy machine) is balanced against a missed failure (downtime). |
| **PR-AUC** | Area under Precision-Recall curve. The gold standard for imbalanced classification because it evaluates the model in the low-recall (rare event) region. |
| **ROC-AUC** | Supplementary metric. Tells us how well the model separates classes across all thresholds. |
| **Hamming Loss** | Multi-label metric: fraction of incorrect label predictions. |
| **Exact Match** | All 5 failure types predicted correctly simultaneously. |

---

## 📊 Results Summary

### Binary Classification — Machine Failure

| Model | CV-F1 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-------|-----------|--------|-----|---------|--------|
| **★ Gradient Boosting** | **0.813** | **0.917** | **0.809** | **0.859** | **0.969** | **0.889** |
| Random Forest | 0.681 | 0.534 | 0.927 | 0.677 | 0.966 | 0.851 |
| Decision Tree | 0.527 | 0.361 | 0.912 | 0.517 | 0.957 | 0.867 |
| KNN | 0.400 | 0.913 | 0.309 | 0.462 | 0.916 | 0.662 |
| Logistic Regression | 0.279 | 0.178 | 0.882 | 0.296 | 0.932 | 0.451 |

### Multi-Label — Individual Failure Types

| Failure | Rate | F1 | PR-AUC | Verdict |
|---------|------|-----|--------|---------|
| **PWF** — Power Failure | 0.95% | **1.000** | **1.000** | ✅ Perfect — clear electrical signature |
| **HDF** — Heat Dissipation | 1.15% | **0.964** | **0.997** | ✅ Near-perfect |
| **OSF** — Overstrain | 0.98% | **0.857** | **0.921** | ✅ Strong |
| **TWF** — Tool Wear | 0.46% | 0.084 | 0.056 | ⚠️ Very rare; needs rule-based fallback |
| **RNF** — Random | 0.19% | 0.000 | 0.002 | ❌ Truly random — cannot predict |

### Confusion Matrix (Best Model — Gradient Boosting)

```
                      Predicted Neg    Predicted Pos
          Actual Neg    1927 (99.7%)        5 (0.3%)
          Actual Pos      13 (19.1%)       55 (80.9%)
```

> **80.9% of actual failures caught with only 5 false alarms per 1,932 non-failures.**

---

## 💡 Key Insights

### 1. 🎯 Three of five failure types are highly predictable
**HDF, PWF, and OSF** can be detected with F1 > 0.85. A real-time monitoring system using **Gradient Boosting** can predict these reliably, enabling preventive maintenance that reduces unplanned downtime by **80–90%** for these categories.

### 2. 🔧 Sensor priority: Torque is king
The **torque_per_speed** ratio is the single most predictive feature (ANOVA F-score: **349.7**), followed by raw torque (299.7). **Product type has near-zero predictive power** (F-score: 1.52, p=0.22).

### 3. ⏰ Tool wear failures have a threshold effect
TWF occurs exclusively at high wear (198–253 min, mean: 216 min vs normal: 107 min). A **two-stage approach** is recommended: monitor wear threshold, then alert when torque/speed ratio deviates at high wear.

### 4. 🎲 Random failures are irreducible
RNF (19 samples) shows **zero correlation** with any sensor reading — these are truly stochastic events that no model can predict.

### 5. 📊 Imbalance handling matters
Standard accuracy is **misleading** at 3.4% failure rate (a "predict never" baseline achieves 96.6%). Always use **PR-AUC and F1-score** for rare-event classification.

---

## 🚢 Production Deployment

### Recommended Architecture

```python
# Pseudocode for real-time monitoring system
monitor = PredictiveMaintenanceSystem(
    model_path="models/gradient_boosting.pkl",
    failure_thresholds={
        'overall': 0.37,    # Optimized F1 threshold
        'HDF': 0.50,
        'PWF': 0.50,
        'OSF': 0.45,
        'TWF': 0.70         # Higher threshold due to rarity
    }
)

# Real-time inference
sensor_reading = {
    'air_temp': 298.5,
    'process_temp': 310.2,
    'speed': 1550,
    'torque': 42.3,
    'tool_wear': 120
}
alert = monitor.predict(sensor_reading)
```

### System Requirements
- Python 3.10+
- 4 GB RAM minimum (8 GB recommended for training)
- OS: Linux, macOS, or Windows

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **AI4I 2020** — The original dataset providers
- **scikit-learn** — Machine learning library
- **imbalanced-learn** — SMOTE/ADASYN implementations
- **matplotlib / seaborn** — Visualization libraries

---

<div align="center">
  <p>
    <strong>Made with ❤️ by the ML Pipeline Team</strong><br/>
    <em>ML Pipeline Project</em>
  </p>
</div>

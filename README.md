# Workflow Overview

This repository contains a generalized machine learning workflow in **R**, demonstrating data preparation, feature selection, model training, anomaly detection, and explainability. Each step is modular, with its own script.

---

## 1. Data Preparation
**Files:**
- `data_prep.R` — direct feature workflow  
- `pca_preprocessing.R` — dimension reduction with PCA  

**Output:**  
Training and test datasets (`train_df`, `test_df`) in the format `[id | features | target]`.  

**Flexibility:**  
Works with both numeric (regression) and categorical (classification) targets.  

---

## 2. Feature Selection (RFE)
**Files:**
- `feature_selection_rfe.R` — feature selection on direct features  
- `pca_rfe_feature_selection.R` — feature selection on PCA-transformed features  

**Method:**  
Recursive Feature Elimination (RFE) with Random Forests.  

**Output:**  
Reduced feature set for model training.  

---

## 3. Model Training
**Files:**
- `model_training.R` — standard model training  
- `model_training_calibrated.R` — model training with post-hoc calibration (regression)  

**Method:**  
Random Forests via `ranger`, with 5-fold CV.  

**Target Types:**  
- **Regression** → reports RMSE, R², MAE  
- **Classification** → reports Accuracy, Kappa, confusion matrix  

**Outputs:**  
- Variable importance plots (`vip`)  
- Saved model objects (`.rds`)  

---

## 4. Anomaly Detection
**Helper Files:**
- `anomaly_detection_uncalibrated.R` — uncalibrated residual analysis  
- `anomaly_detection_calibrated.R` — residual analysis with calibration (regression)  
- `anomaly_detection_classifier` — residual analysis for classifiers  

**Execution Files:**
- `run_anomaly_detection.R` — run anomaly detection for regression  
- `run_anomaly_detection_classifier` — run anomaly detection for classification  

**Methods:**  
Conformal thresholds and MAD-based z-score cutoffs.  

**Outputs:**  
- Results tables with anomaly flags  
- Residual plots  

---

## 5. Explainability (LIME)
**File:**
- `lime_analysis.R`  

**Method:**  
LIME (Local Interpretable Model-agnostic Explanations) to explain predictions.  

**Output:**  
Local feature contribution plots for test set observations.  

---

## Key Features
- **Generalized**: Portable to any dataset by adjusting feature and target definitions.  
- **Flexible**: Regression and classification supported, with or without PCA.  
- **Explainable**: Combines global (VIP) and local (LIME) interpretability.  
- **Practical**: Includes anomaly detection helpers for model monitoring.  

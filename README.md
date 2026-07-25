# 🫁 SmokeSense AI — Smoker Status Prediction (SVM + PCA)

An end-to-end machine learning project that predicts whether a person is a smoker based on bio-signal health data, using **PCA for dimensionality reduction** and **Support Vector Machines (SVM)** for classification. Built on the Kaggle Playground Series **S3E24 — Binary Prediction of Smoker Status using Bio-Signals** dataset, and shipped with an interactive **Streamlit** dashboard.

## 📌 Overview

This repository contains:
- A full data science notebook covering EDA, preprocessing, PCA, and SVM modeling.
- A polished, multi-page **Streamlit web app** (`app.py`) for exploring the data and testing live predictions.
- The training dataset (`train.csv`) used throughout the analysis.

**Goal:** Predict smoking status (smoker / non-smoker) from 22 bio-signal features such as blood pressure, cholesterol, hemoglobin, liver enzymes, and more.

## 🗂️ Project Roadmap

1. **Environment Setup** — import libraries and configure the workspace.
2. **Train / Test Split** — stratified 80/20 split performed *before* any preprocessing to avoid data leakage.
3. **Exploratory Data Analysis** — distribution plots, KDE plots, bar charts, and a correlation heatmap (train set only).
4. **Data Cleaning & Preprocessing** — median imputation followed by feature scaling via a `Pipeline`/`ColumnTransformer`.
5. **Dimensionality Reduction (PCA)** — reduces 22 raw features down to 17 principal components while retaining ~95% of the variance.
6. **Hard vs. Soft Margin SVM** — comparing `LinearSVC` with `C=1.0` (soft margin) against `C=1e6` (hard margin), with the decision boundary visualized on a 2D PCA projection.
7. **Model Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, and ROC curve/AUC.

## 📊 Dataset

- **Source:** Kaggle Playground Series S3E24 (Bio-Signals Smoking Dataset)
- **Records:** 159,256
- **Features:** 22 bio-signal measurements, including age, height, weight, waist circumference, eyesight, hearing, blood pressure, fasting blood sugar, cholesterol (total/HDL/LDL), triglycerides, hemoglobin, urine protein, serum creatinine, liver enzymes (AST, ALT, GTP), and dental caries.
- **Target:** `smoking` (0 = Non-Smoker, 1 = Smoker)

### Key EDA Findings
| Biomarker | Smokers | Non-Smokers | Insight |
|---|---|---|---|
| Triglycerides | 152.54 mg/dL (mean) | 108.20 mg/dL | Smokers cross into the "borderline high" range |
| GTP / γ-GT | 37.0 U/L (median) | 21.0 U/L | ~76% higher liver stress in smokers |
| Hemoglobin | Right-shifted distribution | Lower distribution | Compensation for CO-induced hypoxia |
| Dental Caries | 22.7% prevalence | 14.0% prevalence | Nicotine degrades enamel and reduces saliva flow |

## 🧠 Model Pipeline

```
Raw Input → SimpleImputer(median) → StandardScaler → PCA(n=17, ~95% variance) → LinearSVC → {0, 1}
```

### Results
| Metric | Soft-SVM (C=1.0) | Hard-SVM (C=1e6) |
|---|---|---|
| Accuracy | 74% | 74% |
| Precision (Smoker) | 0.69 | — |
| Recall (Smoker) | 0.75 | — |
| F1-Score (Smoker) | 0.72 | — |
| ROC-AUC | ~0.81 | — |

## 🖥️ Streamlit App

The app (`app.py`) includes four pages:
- **🏠 Home** — project overview, key stats, and the phase roadmap.
- **📊 EDA Dashboard** — interactive Plotly charts (class distribution, liver enzymes, hemoglobin, weight/waist, correlation heatmap).
- **🗂️ Dataset Explorer** — filterable data table with CSV download.
- **🤖 Smart Prediction** — a bio-signal input form that runs live predictions through the trained SVM pipeline.

> **Note:** The interactive dashboard/prediction demo runs on a synthetic dataset statistically calibrated to match the real notebook's numbers, while all displayed metrics (accuracy, AUC, medians, etc.) come directly from the real 159K-record analysis.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+

### Installation
```bash
git clone https://github.com/Shimaasaid999/smoking-analysis-svm.git
cd smoking-analysis-svm
pip install -r requirements.txt
```

### Run the notebook
Open `smoker_svm_notebook_final.ipynb` in Jupyter to walk through the full analysis, from EDA to model evaluation.

### Run the Streamlit app
```bash
streamlit run app.py
```
Then open the local URL shown in your terminal (usually `http://localhost:8501`).

## 🛠️ Tech Stack
- **Python**, **NumPy**, **Pandas**
- **scikit-learn** (SVM, PCA, preprocessing, pipelines, metrics)
- **Streamlit** (web app)
- **Plotly** (interactive visualizations)

## 📁 Repository Structure
```
smoking-analysis-svm/
├── app.py                          # Streamlit dashboard & prediction app
├── smoker_svm_notebook_final.ipynb # Full EDA + modeling notebook
├── train.csv                       # Training dataset
└── requirements.txt                # Python dependencies
```

## ⚠️ Disclaimer
This project is for **educational and research purposes only**. It is not intended for medical diagnosis or clinical decision-making.

## 👤 Author
**Shimaa Said**
GitHub: [@Shimaasaid999](https://github.com/Shimaasaid999)

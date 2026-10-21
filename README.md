# Kaggle Competitions

### Competitive Machine Learning Pipelines, Feature Engineering, and Stacking Ensembles

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.3%2B-276DC3.svg)](https://www.r-project.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f89939.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-268B04.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB6324.svg)](https://xgboost.readthedocs.io/)
[![Tidymodels](https://img.shields.io/badge/Tidymodels-R-blue.svg)](https://www.tidymodels.org/)

🌐 **Language / Langue** : **English 🇬🇧** • [Français 🇫🇷](README.fr.md)

---

A structured portfolio of end-to-end Machine Learning solutions for **Kaggle competitions**, featuring advanced tabular preprocessing, leak-free out-of-fold cross-validation, and multi-model ensembling across both **Python** and **R**.

---

## Features

In competitive data science, off-the-shelf baseline algorithms often suffer from overfitting, high variance, and suboptimal handling of domain-specific feature distributions.

**Kaggle Competitions** addresses these challenges through structured pipelines:
- **Maximizes predictive accuracy** through 10-Fold Out-Of-Fold (OOF) stacking combining Ridge, Lasso, ElasticNet, SVR, Huber GBR, LightGBM, and XGBoost (**CV RMSE ~ 0.107** on House Prices).

- **Prevents data leakage** using Scikit-Learn `ColumnTransformer` pipelines branching numerical imputation/scaling and categorical one-hot encoding.

- **Provides dual-language implementations** comparing modern Python ML ecosystems with R's `tidymodels` and `glmnet` frameworks.

- **Handles complex data distributions** via log1p target transformations, Box-Cox skewness mitigation, and contextual missingness imputation.

---

## Projects Catalog

| Project | Target / Objective | Core Approach | Key Technologies | Performance |
| :--- | :--- | :--- | :--- | :--- |
| [**House Prices**](./house-prices-advanced-regression/) | Advanced House Price Regression | Dean De Cock outlier filtering, domain feature engineering, skewness correction, 10-Fold OOF multi-model stacking | Python & R Tidymodels | **RMSE ~ 0.107** |
| [**Titanic**](./titanic-survival-prediction/) | Binary Passenger Survival Prediction | Leak-free production pipelines, median/mode imputation, regularized logistic regression | Python & R glmnet | **Acc ~ 0.81** |
| [**Santander**](./santander-customer-satisfaction/) | Imbalanced Customer Dissatisfaction | Stratified K-Fold CV, exploratory data analysis, Decision Trees, and Random Forests | Python, Scikit-Learn | **AUC ~ 0.82** |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/delaubier/kaggle.git
cd kaggle
```

### 2. Python Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install numpy pandas scipy scikit-learn lightgbm xgboost catboost matplotlib seaborn
```

### 3. R Environment Setup
```r
install.packages(c("tidyverse", "tidymodels", "glmnet"))
```

---

## Usage

### Run House Prices Stacking Model (Python)
```bash
cd house-prices-advanced-regression
python model.py
```

### Run House Prices Tidymodels Pipeline (R)
```r
source("house-prices-advanced-regression/model_tidymodels.R")
```

### Run Titanic Pipeline (Python)
```bash
cd titanic-survival-prediction
python model_pipeline.py
```

# Kaggle Competitions

### Machine Learning Pipelines and Competitive Modeling

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.3%2B-276DC3.svg)](https://www.r-project.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-f89939.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-268B04.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB6324.svg)](https://xgboost.readthedocs.io/)

🌐 **Language / Langue** : **English 🇬🇧** • [Français 🇫🇷](README.fr.md)

---

A collection of machine learning solutions developed for Kaggle competitions, focusing on feature engineering, cross-validation, and ensembling in Python and R.

---

## Highlights

- **Multi-model stacking:** 10-fold cross-validation ensembling Ridge, Lasso, SVR, LightGBM, and XGBoost on House Prices (CV RMSE ~ 0.107).
- **Leak-free preprocessing:** Scikit-learn pipelines combining imputation, scaling, and categorical encoding.
- **Dual implementation:** Comparing Python and R Tidymodels workflows on tabular datasets.

---

## Projects

| Project | Task | Key Techniques | Stack |
| :--- | :--- | :--- | :--- |
| [**House Prices**](./house-prices-advanced-regression/) | Regression | Outlier cleaning, feature engineering, 10-fold stacking | Python & R |
| [**Titanic**](./titanic-survival-prediction/) | Binary Classification | Production pipelines, regularized logistic regression | Python & R |
| [**Santander**](./santander-customer-satisfaction/) | Imbalanced Classification | Stratified cross-validation, decision trees, random forests | Python |

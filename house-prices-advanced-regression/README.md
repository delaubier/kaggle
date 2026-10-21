# 🏠 House Prices: Advanced Regression Techniques

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-Tidymodels-276DC3.svg?logo=r&logoColor=white)](https://www.tidymodels.org/)
[![Performance](https://img.shields.io/badge/CV%20RMSE-0.107-success.svg)]()

This project presents a dual implementation (**Python** and **R Tidymodels**) to predict residential house prices in Ames, Iowa, featuring an end-to-end Machine Learning workflow with domain-specific feature engineering, out-of-fold cross-validation, and multi-model ensembling.

---

## 🔬 Pipeline Architecture

```
Raw Data ─► Outlier Removal ─► log1p Target ─► Domain Imputation ─► Feature Engineering
                                                                          │
  ┌───────────────────────────────────────────────────────────────────────┘
  ▼
Skewness Reduction (log1p) ─► RobustScaler & One-Hot Encoding
  │
  ├─► RidgeCV / LassoCV / ElasticNetCV
  ├─► Support Vector Regressor (SVR)
  ├─► Gradient Boosting Regressor (Huber Loss) ──► 10-Fold OOF Predictions ──► Stacking Ensemble
  ├─► LightGBM Regressor
  └─► XGBoost Regressor
```

### Key Engineering Steps:
1. **Outlier Filtering:** Removal of extreme outliers based on official Dean De Cock recommendations (`GrLivArea > 4000 sq ft` with `SalePrice < $300k`).
2. **Log-Transform Target:** Aligns optimization directly with the competition's RMSLE metric: $\tilde{y} = \log(1 + y)$.
3. **Contextual Imputation:** Missing values in amenities (garage, basement, pool, fireplace) are imputed contextually as absence (`"None"` or `0`).
4. **Domain Feature Engineering:**
   - Composite surface metrics: `TotalSF`, `TotalPorchSF`.
   - Weighted plumbing variables: `TotalBaths`.
   - Property timeline: age at sale, time since remodeling, garage vintage.
   - Interaction terms: `OverallGrade`, `TotalSF * OverallQual`, `GrLivArea / TotalSF`.
5. **Ensemble Architecture:**
   - Out-Of-Fold (OOF) cross-validation across 7 base learners.
   - Meta-learner weighting to minimize generalization variance.

---

## 💻 Running the Models

### Python Execution
```bash
python model.py
```
Outputs cross-validation folds, individual model RMSEs, and generates `submission.csv`.

### R Tidymodels Execution
```r
source("model_tidymodels.R")
```
Demonstrates modern tidyverse grammar, recipe-based preprocessing, and cross-validated modeling.

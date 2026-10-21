# 🚢 Titanic: Machine Learning from Disaster

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/c/titanic)
[![Python](https://img.shields.io/badge/Python-Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![R](https://img.shields.io/badge/R-glmnet-276DC3.svg?logo=r&logoColor=white)](https://cran.r-project.org/web/packages/glmnet/index.html)

A structured classification pipeline tackling binary survival prediction on the famous Titanic passenger manifest. Implemented in both **Python** (Scikit-Learn Pipelines) and **R** (regularized generalized linear models).

---

## 🛠️ Methodological Highlights

- **Production Scikit-Learn Pipeline:** Uses `ColumnTransformer` to seamlessly branch numerical preprocessing (median imputation + `StandardScaler`) and categorical preprocessing (mode imputation + `OneHotEncoder(drop='first', handle_unknown='ignore')`), preventing data leakage across train and test folds.
- **R `glmnet` Regularization:** Ridge/Lasso/ElasticNet logistic regression with log-transformed `Fare` features and categorical factor encoding.

## 🚀 Usage

### Python Pipeline
```bash
python model_pipeline.py
```

### R Script
```r
source("model_glmnet.R")
```

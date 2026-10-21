# 🏆 Kaggle Competitions Portfolio

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.3%2B-276DC3.svg?logo=r&logoColor=white)](https://www.r-project.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-informational.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB6324.svg)](https://xgboost.readthedocs.io/)
[![Tidymodels](https://img.shields.io/badge/Tidymodels-R-blue.svg)](https://www.tidymodels.org/)

A curated collection of competitive machine learning solutions developed for **Kaggle competitions**. Demonstrates advanced feature engineering, out-of-fold validation, gradient boosting, stacking ensembles, and reproducible pipelines across both **Python** and **R**.

---

## 📊 Summary of Projects

| Competition | Objective | Methodologies & Models | Key Tech | CV Score |
| :--- | :--- | :--- | :--- | :--- |
| [**House Prices**](./house-prices-advanced-regression/) | Sale Price Regression | Dean De Cock outlier removal, Skewness correction, Domain imputation, 10-Fold OOF Stacking (Ridge, Lasso, ElasticNet, SVR, Huber GBR, LightGBM, XGBoost) | Python & R Tidymodels | **RMSE ~ 0.107** |
| [**Titanic**](./titanic-survival-prediction/) | Binary Classification | Production Pipeline (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`), Regularized Logistic Regression (`glmnet`) | Python & R | **Acc ~ 0.81** |
| [**Santander**](./santander-customer-satisfaction/) | Imbalanced Classification | Exploratory Data Analysis, Stratified K-Fold CV, Decision Trees, Random Forests, ROC/AUC assessment | Python | **AUC ~ 0.82** |

---

## 🛠️ Repository Structure

```
kaggle/
├── house-prices-advanced-regression/
│   ├── model.py                  # Python SOTA Stacking & Multi-model pipeline
│   ├── model_tidymodels.R        # R Tidymodels implementation
│   ├── data/                     # Train, test, data description
│   └── README.md                 # Detailed methodology & pipeline breakdown
│
├── titanic-survival-prediction/
│   ├── model_pipeline.py         # Scikit-learn Pipeline with ColumnTransformer
│   ├── model_glmnet.R            # R glmnet regularized regression
│   ├── data/                     # Dataset and submission samples
│   └── README.md                 # Modeling notes & metrics
│
└── santander-customer-satisfaction/
    ├── model.ipynb               # EDA, Feature selection & Tree-based models
    ├── data/                     # Metadata & sample submission
    └── README.md                 # Overview & Kaggle dataset setup
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/kaggle.git
cd kaggle
```

### 2. Python Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy pandas scipy scikit-learn lightgbm xgboost catboost matplotlib seaborn
```

### 3. R Environment Setup
```r
install.packages(c("tidyverse", "tidymodels", "glmnet"))
```

---

## 👤 Author

**Thomas de Laubier**  
- Master's in Applied Mathematics & Statistics, Data Science & AI (*Université Rennes 2*)
- [LinkedIn Profile](https://www.linkedin.com) • [GitHub Profile](https://github.com)


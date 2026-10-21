# Compétitions Kaggle

### Pipelines de Machine Learning et modélisation compétitive

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.3%2B-276DC3.svg)](https://www.r-project.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-f89939.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-268B04.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB6324.svg)](https://xgboost.readthedocs.io/)

🌐 **Langue / Language** : [English 🇬🇧](README.md) • **Français 🇫🇷**

---

Un ensemble de solutions de Machine Learning pour les compétitions Kaggle, axé sur le feature engineering, la validation croisée et l'ensembling en Python et R.

---

## Points clés

- **Stacking multi-modèles :** Validation croisée 10-fold combinant Ridge, Lasso, SVR, LightGBM et XGBoost sur House Prices (CV RMSE ~ 0.107).
- **Pipelines hermétiques :** Pipelines Scikit-learn associant imputation, normalisation et encodage sans fuite de données.
- **Double approche :** Comparaison des écosystèmes Python et R Tidymodels sur données tabulaires.

---

## Projets

| Projet | Tâche | Méthodologie | Technologies |
| :--- | :--- | :--- | :--- |
| [**House Prices**](./house-prices-advanced-regression/) | Régression | Nettoyage, feature engineering, stacking 10-fold | Python & R |
| [**Titanic**](./titanic-survival-prediction/) | Classification binaire | Pipelines Scikit-learn, régression logistique régularisée | Python & R |
| [**Santander**](./santander-customer-satisfaction/) | Classification déséquilibrée | Validation stratifiée, arbres de décision, random forests | Python |

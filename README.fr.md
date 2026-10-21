# Compétitions Kaggle

### Pipelines de Machine Learning compétitif, Feature Engineering avancé et Stacking multi-modèles

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.3%2B-276DC3.svg)](https://www.r-project.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f89939.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-268B04.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB6324.svg)](https://xgboost.readthedocs.io/)
[![Tidymodels](https://img.shields.io/badge/Tidymodels-R-blue.svg)](https://www.tidymodels.org/)

🌐 **Langue / Language** : [English 🇬🇧](README.md) • **Français 🇫🇷**

---

Un portfolio structuré de solutions complètes de Machine Learning développées pour les **compétitions Kaggle**, intégrant du prétraitement tabulaire avancé, une validation croisée sans fuite de données (*out-of-fold*), et des architectures d'ensembling en **Python** et **R**.

---

## Fonctionnalités

En science des données compétitive, les modèles bruts souffrent fréquemment de surapprentissage, d'une variance élevée et d'une prise en compte médiocre des distributions de variables métier.

**Compétitions Kaggle** répond à ces exigences par des architectures éprouvées :
- **Maximise la performance prédictive** grâce à un stacking 10-Fold Out-Of-Fold combinant Ridge, Lasso, ElasticNet, SVR, Huber GBR, LightGBM et XGBoost (**CV RMSE ~ 0.107** sur House Prices).

- **Élimine toute fuite d'information (*data leakage*)** via des pipelines Scikit-Learn `ColumnTransformer` séparant imputations et encodages catégoriels.

- **Propose une double implémentation Python & R** comparant les écosystèmes modernes de Machine Learning avec `tidymodels` et `glmnet`.

- **Traite les distributions asymétriques** par transformation log1p de la cible, correction de skewness et imputation contextuelle des valeurs manquantes.

---

## Catalogue des projets

| Projet | Cible / Objectif | Approche clé | Technologies | Performance |
| :--- | :--- | :--- | :--- | :--- |
| [**House Prices**](./house-prices-advanced-regression/) | Régression avancée des prix immobiliers | Filtrage d'outliers (Dean De Cock), feature engineering métier, réduction d'asymétrie, stacking 10-Fold OOF de 7 modèles | Python & R Tidymodels | **RMSE ~ 0.107** |
| [**Titanic**](./titanic-survival-prediction/) | Classification binaire de survie | Pipelines hermétiques Scikit-Learn, imputation médiane/mode, régression logistique régularisée | Python & R glmnet | **Acc ~ 0.81** |
| [**Santander**](./santander-customer-satisfaction/) | Détection de clients insatisfaits (fort déséquilibre) | Validation croisée stratifiée K-Fold, analyse exploratoire, arbres de décision et Random Forests | Python, Scikit-Learn | **AUC ~ 0.82** |

---

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/delaubier/kaggle.git
cd kaggle
```

### 2. Environnement Python
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install numpy pandas scipy scikit-learn lightgbm xgboost catboost matplotlib seaborn
```

### 3. Environnement R
```r
install.packages(c("tidyverse", "tidymodels", "glmnet"))
```

---

## Utilisation

### Exécuter le modèle Stacking House Prices (Python)
```bash
cd house-prices-advanced-regression
python model.py
```

### Exécuter le pipeline Tidymodels House Prices (R)
```r
source("house-prices-advanced-regression/model_tidymodels.R")
```

### Exécuter le pipeline Titanic (Python)
```bash
cd titanic-survival-prediction
python model_pipeline.py
```

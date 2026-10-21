# 💳 Santander Customer Satisfaction

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/c/santander-customer-satisfaction)
[![Python](https://img.shields.io/badge/Python-Jupyter-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Ensemble-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An exploratory data analysis and supervised classification workflow for identifying dissatisfied banking customers early in their relationship with Santander Bank.

---

## 🔍 Methodology

- **Class Imbalance Handling:** The target variable (`TARGET = 1` for unhappy customers) is heavily imbalanced (~4% positive rate).
- **Validation Scheme:** Stratified K-Fold cross-validation to ensure reliable ROC/AUC evaluation across all splits.
- **Tree-Based Models:** Exploration of single Decision Trees with depth pruning and Random Forests to establish a baseline.

## 📦 Data Download Instructions

Due to GitHub's file size limit (raw training data is >300 MB uncompressed), datasets should be downloaded directly via Kaggle CLI:

```bash
kaggle competitions download -c santander-customer-satisfaction
unzip santander-customer-satisfaction.zip -d data/
```

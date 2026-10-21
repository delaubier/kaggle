"""
Kaggle House Prices: Advanced Regression Techniques
State-of-the-Art Ensembling Model (CV RMSE ~ 0.107)

Pipeline:
1. Détection et suppression des outliers extrêmes (Dean De Cock, GrLivArea > 4000 sq ft avec SalePrice < $300k).
2. Transformation logarithmique de la cible (log1p) pour alignement direct avec la métrique RMSLE.
3. Imputation métier contextuelle des valeurs manquantes (NA = absence de garage/sous-sol/piscine/cheminée).
4. Encodage ordinal précis des variables de notation et qualité (Po, Fa, TA, Gd, Ex).
5. Feature Engineering avancé :
   - Surfaces totales combinées (TotalSF, TotalPorchSF).
   - Sanitaires totaux pondérés (TotalBaths).
   - Âge du bien, rénovations et ancienneté du garage.
   - Indicateurs binaires de présence d'équipements clés.
   - Interactions multi-critères (OverallGrade, TotalSF * Qualité, GrLivArea / TotalSF, etc.).
6. Réduction de l'asymétrie (skewness) par log1p sur les variables numériques fortement asymétriques.
7. Encodage One-Hot des variables catégorielles avec alignement train/test et RobustScaler.
8. Validation croisée 10-Fold avec prédictions Out-Of-Fold (OOF) sur un ensemble diversifié de modèles SOTA :
   - RidgeCV
   - LassoCV
   - ElasticNetCV
   - Support Vector Regressor (SVR)
   - Gradient Boosting Regressor (Huber loss)
   - LightGBM Regressor
   - XGBoost Regressor
   - CatBoost Regressor
9. Optimisation sous contraintes (SLSQP) des poids de blending sur les prédictions OOF.
10. Prédictions test par bagging 10-fold et inversion expm1 pour soumission finale.
"""

import copy
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skew
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")


def main():
    start_time = time.time()
    print("=" * 65)
    print(" KAGGLE HOUSE PRICES - HIGH PERFORMANCE ENSEMBLE PIPELINE")
    print("=" * 65)

    # ---------------------------------------------------------
    # 1. Chargement des données
    # ---------------------------------------------------------
    print("\n[1/7] Chargement des données...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    print(f"  Dimensions initiales - Train : {train.shape}, Test : {test.shape}")

    # ---------------------------------------------------------
    # 2. Suppression des outliers (Recommandation officielle Dean De Cock)
    # ---------------------------------------------------------
    print("\n[2/7] Traitement des outliers...")
    outliers = train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)].index
    print(f"  Suppression de {len(outliers)} outliers identifiés (IDs : {train.loc[outliers, 'Id'].tolist()})")
    train = train.drop(outliers).reset_index(drop=True)

    # Transformation logarithmique de la variable cible
    y = np.log1p(train["SalePrice"]).values
    n_train = len(train)
    test_ids = test["Id"].values

    # Concaténation pour un prétraitement cohérent
    full_df = pd.concat(
        [train.drop(columns=["Id", "SalePrice"]), test.drop(columns=["Id"])],
        axis=0,
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Imputation métier des valeurs manquantes
    # ---------------------------------------------------------
    print("\n[3/7] Imputation des valeurs manquantes & encodages métier...")

    # Catégorielles où NA signifie absence d'équipement ("None")
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType"
    ]
    for col in none_cols:
        full_df[col] = full_df[col].fillna("None")

    # Numériques où NA signifie absence / 0
    zero_cols = [
        "GarageCars", "GarageArea", "GarageYrBlt",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
    ]
    for col in zero_cols:
        full_df[col] = full_df[col].fillna(0)

    # LotFrontage : médiane par quartier (Neighborhood)
    full_df["LotFrontage"] = full_df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda s: s.fillna(s.median())
    )
    full_df["LotFrontage"] = full_df["LotFrontage"].fillna(full_df["LotFrontage"].median())

    # Variables catégorielles résiduelles : imputation par le mode
    mode_cols = ["MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType", "Utilities"]
    for col in mode_cols:
        full_df[col] = full_df[col].fillna(full_df[col].mode()[0])

    full_df["Functional"] = full_df["Functional"].fillna("Typ")

    # Conversion de variables numériques en chaînes de caractères catégorielles
    full_df["MSSubClass"] = full_df["MSSubClass"].astype(str)
    full_df["MoSold"] = full_df["MoSold"].astype(str)
    full_df["YrSold_cat"] = full_df["YrSold"].astype(str)

    # Encodages ordinaux explicites
    qual_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    for col in [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
        "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"
    ]:
        full_df[col + "_Num"] = full_df[col].map(qual_map).fillna(0).astype(int)

    full_df["BsmtExposure_Num"] = full_df["BsmtExposure"].map(
        {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
    ).fillna(0).astype(int)

    fin_map = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    full_df["BsmtFinType1_Num"] = full_df["BsmtFinType1"].map(fin_map).fillna(0).astype(int)
    full_df["BsmtFinType2_Num"] = full_df["BsmtFinType2"].map(fin_map).fillna(0).astype(int)

    func_map = {"None": 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}
    full_df["Functional_Num"] = full_df["Functional"].map(func_map).fillna(0).astype(int)

    full_df["GarageFinish_Num"] = full_df["GarageFinish"].map(
        {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
    ).fillna(0).astype(int)

    full_df["Fence_Num"] = full_df["Fence"].map(
        {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}
    ).fillna(0).astype(int)

    full_df["CentralAir_Num"] = (full_df["CentralAir"] == "Y").astype(int)

    # ---------------------------------------------------------
    # 4. Feature Engineering
    # ---------------------------------------------------------
    print("\n[4/7] Ingénierie des caractéristiques (Feature Engineering)...")

    # Agrégation des surfaces
    full_df["TotalSF"] = full_df["TotalBsmtSF"] + full_df["1stFlrSF"] + full_df["2ndFlrSF"]
    full_df["TotalPorchSF"] = (
        full_df["WoodDeckSF"] + full_df["OpenPorchSF"] +
        full_df["EnclosedPorch"] + full_df["3SsnPorch"] + full_df["ScreenPorch"]
    )
    full_df["TotalBaths"] = (
        full_df["FullBath"] + 0.5 * full_df["HalfBath"] +
        full_df["BsmtFullBath"] + 0.5 * full_df["BsmtHalfBath"]
    )

    # Temporalité et rénovations
    yr_sold = full_df["YrSold"].astype(int)
    full_df["HouseAge"] = np.maximum(0, yr_sold - full_df["YearBuilt"])
    full_df["RemodAge"] = np.maximum(0, yr_sold - full_df["YearRemodAdd"])
    garage_yr = np.where(full_df["GarageYrBlt"] == 0, full_df["YearBuilt"], full_df["GarageYrBlt"])
    full_df["GarageAge"] = np.maximum(0, yr_sold - garage_yr)
    full_df["IsRemodeled"] = (full_df["YearRemodAdd"] != full_df["YearBuilt"]).astype(int)
    full_df["IsNew"] = (yr_sold == full_df["YearBuilt"]).astype(int)

    # Drapeaux de présence binaire
    full_df["HasPool"] = (full_df["PoolArea"] > 0).astype(int)
    full_df["HasGarage"] = (full_df["GarageArea"] > 0).astype(int)
    full_df["HasBsmt"] = (full_df["TotalBsmtSF"] > 0).astype(int)
    full_df["HasFireplace"] = (full_df["Fireplaces"] > 0).astype(int)
    full_df["Has2ndFlr"] = (full_df["2ndFlrSF"] > 0).astype(int)
    full_df["HasWoodDeck"] = (full_df["WoodDeckSF"] > 0).astype(int)
    full_df["HasOpenPorch"] = (full_df["OpenPorchSF"] > 0).astype(int)
    full_df["HasEnclosedPorch"] = (full_df["EnclosedPorch"] > 0).astype(int)

    # Interactions & Ratios
    full_df["OverallGrade"] = full_df["OverallQual"] * full_df["OverallCond"]
    full_df["TotalSF_OverallQual"] = full_df["TotalSF"] * full_df["OverallQual"]
    full_df["ExterGrade"] = full_df["ExterQual_Num"] * full_df["ExterCond_Num"]
    full_df["KitchenGrade"] = full_df["KitchenQual_Num"] * full_df["KitchenAbvGr"]
    full_df["GarageGrade"] = full_df["GarageQual_Num"] * full_df["GarageCond_Num"]

    full_df["TotalQual"] = (
        full_df["OverallQual"] + full_df["OverallCond"] +
        full_df["ExterQual_Num"] + full_df["ExterCond_Num"] +
        full_df["BsmtQual_Num"] + full_df["BsmtCond_Num"] +
        full_df["HeatingQC_Num"] + full_df["KitchenQual_Num"] +
        full_df["FireplaceQu_Num"] + full_df["GarageQual_Num"] + full_df["GarageCond_Num"]
    )
    full_df["SF_per_Room"] = full_df["TotalSF"] / (full_df["TotRmsAbvGrd"] + full_df["TotalBaths"] + 1)
    full_df["GrLivArea_ratio"] = full_df["GrLivArea"] / (full_df["TotalSF"] + 1)
    full_df["BsmtFinRatio"] = (full_df["BsmtFinSF1"] + full_df["BsmtFinSF2"]) / (full_df["TotalBsmtSF"] + 1)

    # ---------------------------------------------------------
    # 5. Normalisation de la skewness et One-Hot Encoding
    # ---------------------------------------------------------
    print("\n[5/7] Traitement de l'asymétrie (skewness) & One-Hot Encoding...")
    num_features = full_df.select_dtypes(include=[np.number]).columns
    skewed_feats = full_df[num_features].apply(lambda s: skew(s.dropna())).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.75].index

    for feat in high_skew:
        full_df[feat] = np.log1p(np.maximum(0, full_df[feat]))

    full_df = pd.get_dummies(full_df, drop_first=True)

    X = full_df.iloc[:n_train].copy().values
    X_test = full_df.iloc[n_train:].copy().values

    # Scaling robuste pour les modèles linéaires & SVM
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Dimensions finales - Train : {X.shape}, Test : {X_test.shape}")

    # ---------------------------------------------------------
    # 6. Modélisation & Validation Croisée 10-Fold
    # ---------------------------------------------------------
    print("\n[6/7] Entraînement et validation croisée 10-Fold (OOF)...")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Définition des estimateurs SOTA
    model_definitions = {
        "Ridge": (
            RidgeCV(alphas=[0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0], cv=5),
            True,  # utilise X_scaled
        ),
        "Lasso": (
            LassoCV(alphas=[0.0001, 0.0003, 0.0005, 0.0007, 0.001], max_iter=10000, cv=5, random_state=42),
            True,
        ),
        "ElasticNet": (
            ElasticNetCV(alphas=[0.0001, 0.0005, 0.001], l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], max_iter=10000, cv=5, random_state=42),
            True,
        ),
        "SVR": (
            SVR(C=20, max_iter=-1, epsilon=0.008, gamma=0.0003),
            True,
        ),
        "GBR": (
            GradientBoostingRegressor(
                n_estimators=1000, learning_rate=0.03, max_depth=3,
                max_features="sqrt", min_samples_leaf=15, min_samples_split=10,
                loss="huber", random_state=42
            ),
            False,
        ),
        "LightGBM": (
            lgb.LGBMRegressor(
                objective="regression", num_leaves=15, learning_rate=0.03,
                n_estimators=1000, max_bin=200, bagging_fraction=0.75,
                bagging_freq=5, feature_fraction=0.4, verbose=-1, random_state=42
            ),
            False,
        ),
        "XGBoost": (
            xgb.XGBRegressor(
                learning_rate=0.03, n_estimators=1000, max_depth=3,
                min_child_weight=1.5, subsample=0.7, colsample_bytree=0.4,
                reg_alpha=0.6, reg_lambda=0.6, random_state=42, n_jobs=-1
            ),
            False,
        ),
        "CatBoost": (
            cb.CatBoostRegressor(
                iterations=1200, learning_rate=0.03, depth=4,
                l2_leaf_reg=3, loss_function="RMSE", verbose=0, random_seed=42
            ),
            False,
        ),
    }

    oof_preds = {name: np.zeros(n_train) for name in model_definitions}
    test_preds_folds = {name: np.zeros(len(X_test)) for name in model_definitions}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        y_train_f, y_val_f = y[train_idx], y[val_idx]

        for name, (base_model, use_scaled) in model_definitions.items():
            X_tr = X_scaled[train_idx] if use_scaled else X[train_idx]
            X_va = X_scaled[val_idx] if use_scaled else X[val_idx]
            X_te = X_test_scaled if use_scaled else X_test

            model_fold = copy.deepcopy(base_model)
            model_fold.fit(X_tr, y_train_f)

            pred_val = model_fold.predict(X_va)
            oof_preds[name][val_idx] = pred_val
            test_preds_folds[name] += model_fold.predict(X_te) / kf.n_splits

    # Affichage des scores individuels OOF
    print("\n  --- Performances OOF individuelles (RMSE log) ---")
    for name in model_definitions:
        score = np.sqrt(np.mean((oof_preds[name] - y) ** 2))
        print(f"  {name:<12}: RMSE = {score:.4f}")

    # ---------------------------------------------------------
    # 7. Optimisation du Blending d'ensemble & Prédictions finales
    # ---------------------------------------------------------
    print("\n[7/7] Optimisation du Blending d'Ensemble & Génération du Submission...")
    model_names = list(model_definitions.keys())
    oof_matrix = np.column_stack([oof_preds[name] for name in model_names])
    test_matrix = np.column_stack([test_preds_folds[name] for name in model_names])

    # Optimisation sous contraintes : somme(w) = 1, w_i >= 0
    def objective_rmse(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = np.dot(oof_matrix, w)
        return np.sqrt(np.mean((pred - y) ** 2))

    init_weights = [1.0 / len(model_names)] * len(model_names)
    bounds = [(0, 1) for _ in range(len(model_names))]
    opt_result = minimize(objective_rmse, init_weights, method="SLSQP", bounds=bounds)
    opt_weights = opt_result.x / np.sum(opt_result.x)

    # Calcul de la performance de l'ensemble
    ensemble_oof = np.dot(oof_matrix, opt_weights)
    ensemble_rmse = np.sqrt(np.mean((ensemble_oof - y) ** 2))

    print("\n  --- Poids optimaux du Blending ---")
    for name, w in zip(model_names, opt_weights):
        print(f"  {name:<12}: {w * 100:5.2f} %")
    print(f"\n  >>> SCORE CV GLOBAL DE L'ENSEMBLE (RMSE log) : {ensemble_rmse:.4f} <<<")

    # Prédictions sur le test set (Bagging 10-fold combiné)
    test_log_preds = np.dot(test_matrix, opt_weights)
    test_final_preds = np.expm1(test_log_preds)

    # Vérification d'intégrité
    assert not np.isnan(test_final_preds).any(), "Erreur : présence de NaN dans les prédictions !"
    assert (test_final_preds > 0).all(), "Erreur : prix prédits négatifs ou nuls !"
    assert len(test_final_preds) == 1459, f"Erreur de dimension : {len(test_final_preds)} au lieu de 1459."

    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": test_final_preds
    })
    submission.to_csv("submission.csv", index=False)

    print(f"\n  Fichier de soumission généré : submission.csv ({len(submission)} lignes)")
    print(f"  Prix moyen prédit : ${submission['SalePrice'].mean():,.2f}")
    print(f"  Prix min prédit   : ${submission['SalePrice'].min():,.2f}")
    print(f"  Prix max prédit   : ${submission['SalePrice'].max():,.2f}")
    print(f"  Temps d'exécution total : {time.time() - start_time:.1f} secondes")
    print("=" * 65)


if __name__ == "__main__":
    main()
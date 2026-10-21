library(tidyverse)
library(tidymodels)

# 1. Chargement des données
train <- read_csv("data/train.csv")
test  <- read_csv("data/test.csv")

# 2. Suppression des outliers (recommandation officielle Dean De Cock)
train <- train |> filter(!(GrLivArea > 4000 & SalePrice < 300000))

# Transformation log(1 + y) sur la variable cible
train <- train |> mutate(SalePrice = log1p(SalePrice))

# 3. Séparation et concaténation pour le feature engineering
test_id <- test$Id
y_train <- train$SalePrice

full_data <- bind_rows(train |> select(-SalePrice), test)

# 4. Traitement des valeurs manquantes & Feature Engineering
# Dictionnaire pour les variables ordinales de qualité
qual_levels <- c("None" = 0, "Po" = 1, "Fa" = 2, "TA" = 3, "Gd" = 4, "Ex" = 5)

full_clean <- full_data |>
  # Imputation des variables catégorielles où NA = absence d'équipement
  mutate(across(c(GarageType, GarageFinish, GarageQual, GarageCond,
                  BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2,
                  FireplaceQu, Fence, MiscFeature, Alley), 
                ~ replace_na(.x, "None")),
         
         # Imputation des numériques où NA = absence (surface/quantité = 0)
         across(c(GarageArea, GarageCars, BsmtFinSF1, BsmtFinSF2, 
                  BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath, MasVnrArea), 
                ~ replace_na(.x, 0))) |>
  
  # Imputation de LotFrontage par la médiane par quartier
  group_by(Neighborhood) |>
  mutate(LotFrontage = if_else(is.na(LotFrontage), median(LotFrontage, na.rm = TRUE), LotFrontage)) |>
  ungroup() |>
  mutate(LotFrontage = replace_na(LotFrontage, median(LotFrontage, na.rm = TRUE))) |>
  
  # Feature Engineering
  mutate(
    TotalSF         = TotalBsmtSF + `1stFlrSF` + `2ndFlrSF`,
    TotalBaths      = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath,
    Age             = YrSold - YearBuilt,
    YearsSinceRemod = YrSold - YearRemodAdd,
    TotalPorchSF    = WoodDeckSF + OpenPorchSF + EnclosedPorch + `3SsnPorch` + ScreenPorch,
    
    # Encodage ordinal explicite des principales notes de qualité
    ExterQualNum    = recode(ExterQual, !!!qual_levels, .default = 0),
    KitchenQualNum  = recode(KitchenQual, !!!qual_levels, .default = 0),
    BsmtQualNum     = recode(BsmtQual, !!!qual_levels, .default = 0),
    HeatingQCNum    = recode(HeatingQC, !!!qual_levels, .default = 0)
  ) |>
  
  # Conversion de variables numériques qui sont en réalité catégorielles
  mutate(MSSubClass = as.factor(MSSubClass),
         MoSold     = as.factor(MoSold))

# 5. Séparation Train / Test après nettoyage
n_train <- nrow(train)
train_clean <- full_clean |> slice(1:n_train) |> mutate(SalePrice = y_train)
test_clean  <- full_clean |> slice((n_train + 1):n())

# 6. Pipeline final avec tidymodels (One-Hot, imputation résiduelle, normalisation)
house_recipe <- recipe(SalePrice ~ ., data = train_clean) |>
  update_role(Id, new_role = "ID") |>
  step_rm(PoolQC) |> # Moins de 1 % de valeurs non nulles
  step_impute_mode(all_nominal_predictors()) |> # Derniers NA résiduels
  step_impute_median(all_numeric_predictors()) |>
  step_novel(all_nominal_predictors()) |>      # Gère les catégories inédites dans le test
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_zv(all_predictors())                     # Supprime les colonnes constantes

# Ajustement de la recette et extraction des matrices prêtes pour modélisation
prepared_recipe <- prep(house_recipe, training = train_clean)

X_train_final <- juice(prepared_recipe)
X_test_final  <- bake(prepared_recipe, new_data = test_clean)

# Aperçu
cat("Dimensions train final :", dim(X_train_final), "\n")
cat("Dimensions test final  :", dim(X_test_final), "\n")

glimpse(X_train_final)

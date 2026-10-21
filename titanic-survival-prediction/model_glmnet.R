# Library
library(glmnet)

# Lecture des données
train <- read.csv("data/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("data/test.csv", stringsAsFactors = FALSE)

# Nettoyage des données
## Train
train$Age[is.na(train$Age)] <- mean(train$Age, na.rm = TRUE)
train$Sex      <- factor(train$Sex)
train$Pclass   <- factor(train$Pclass)
train$Survived <- factor(train$Survived)
train$Embarked[is.na(train$Embarked) | train$Embarked == ""] <- "S"
train$Embarked <- factor(train$Embarked, levels = c("C", "Q", "S"))
train$Fare     <- log1p(train$Fare)

## Test
test$Age[is.na(test$Age)]   <- mean(train$Age, na.rm = TRUE)
test$Fare[is.na(test$Fare)] <- median(train$Fare, na.rm = TRUE)
test$Fare     <- log1p(test$Fare)
test$Sex      <- factor(test$Sex, levels = levels(train$Sex))
test$Pclass   <- factor(test$Pclass, levels = levels(train$Pclass))
test$Embarked[is.na(test$Embarked) | test$Embarked == ""] <- "S"
test$Embarked <- factor(test$Embarked, levels = levels(train$Embarked))

# Matrices
df_train <- train[, -c(1, 4, 9, 11)]
df_test  <- test[c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")]

X_train <- model.matrix(Survived ~ (.)^2, data = df_train)[, -1]
X_test  <- model.matrix(~ (.)^2, data = df_test)[, -1]

# Modèle & CV
set.seed(42)
cv_fit <- cv.glmnet(
  X_train, 
  df_train$Survived, 
  family = "binomial", 
  alpha = 1, 
  type.measure = "class", 
  keep = TRUE
)

# Extraction des probabilités hors-échantillon
idx_lambda <- which(cv_fit$lambda == cv_fit$lambda.min)
logits_oof <- cv_fit$fit.preval[, idx_lambda]
proba_oof  <- 1 / (1 + exp(-logits_oof))

# Recherche du seuil optimal
seuils <- seq(0.30, 0.70, by = 0.01)
acc_par_seuil <- sapply(seuils, function(s) {
  preds <- ifelse(proba_oof >= s, 1, 0)
  mean(preds == train$Survived)
})

meilleur_idx   <- which.max(acc_par_seuil)
seuil_optimal  <- seuils[meilleur_idx]
meilleure_acc  <- acc_par_seuil[meilleur_idx]
acc_defaut     <- mean(ifelse(proba_oof >= 0.5, 1, 0) == train$Survived)

cat(sprintf("Accuracy au seuil 0.50 : %.2f %%\n", acc_defaut * 100))
cat(sprintf("Seuil optimal trouvé   : %.2f\n", seuil_optimal))
cat(sprintf("Accuracy maximale (CV) : %.2f %%\n", meilleure_acc * 100))

# Prédictions avec le seuil calculé
test$proba  <- as.vector(predict(cv_fit, newx = X_test, s = "lambda.min", type = "response"))
test$result <- as.vector(ifelse(test$proba >= seuil_optimal, 1, 0))

submission <- data.frame(PassengerId = test$PassengerId, Survived = test$result)
write.csv(submission, "submission_v3.csv", row.names = FALSE)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_num = ["Age", "Fare", "SibSp", "Parch"]
X_cat = ["Pclass", "Sex", "Embarked"]

X_train = train[X_cat + X_num]
X_test = test[X_cat + X_num]
Y_train = train["Survived"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), X_num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ]), X_cat)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(X_train, Y_train)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": model.predict(X_test)
})
submission.to_csv("submission.csv", index=False)
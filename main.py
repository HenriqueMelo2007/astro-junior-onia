# Only the main python code here

# imports
import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# training
trainFile = pandas.read_csv("treino.csv")

X = trainFile.drop("target", axis=1)
y = trainFile["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# models
models = {
    "Optimized Random Forest": RandomForestClassifier(
        random_state=42, n_estimators=500, bootstrap=False
    ),
    "Default Random Forest": RandomForestClassifier(random_state=42),
}


# execution using default params
def fit_and_score(models, X_train, y_train):
    model_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5)
        model_scores[name] = scores.mean()
    return model_scores


model_scores = fit_and_score(models=models, X_train=X_train, y_train=y_train)
print(model_scores)
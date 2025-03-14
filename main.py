# Only the main python code here

# imports
import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# training
trainFile = pandas.read_csv("treino.csv")

X = trainFile.drop("target", axis=1)
y = trainFile["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# models
models = {
    "Optimized Random Forest": RandomForestClassifier(random_state=42, n_estimators=500, bootstrap=False),
    "Default Random Forest": RandomForestClassifier(random_state=42),
}


# execution using default params
def fit_and_score(models, X_train, X_test, y_train, y_test):

    model_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


model_scores = fit_and_score(
    models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)
print(model_scores)
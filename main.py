# Only the main python code here

# imports
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

mlp_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(random_state=42, max_iter=1000, solver='sgd', hidden_layer_sizes=(100, 50), activation='tanh'))
    ]
)

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
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Ada Boost": AdaBoostClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "MLP Pipeline": mlp_pipeline,
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
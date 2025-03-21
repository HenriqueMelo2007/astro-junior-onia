import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.neural_network import MLPClassifier

SEED = 42
np.random.seed(SEED)

train_df = pd.read_csv("treino.csv")
test_df = pd.read_csv("teste.csv")

print("Distribuição das classes no conjunto de treino:")
class_counts = train_df["target"].value_counts().sort_index()
print(class_counts)
print("Percentuais:\n", (class_counts / train_df.shape[0]) * 100)

numeric_features = [col for col in train_df.columns if col not in ["id", "target"]]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                ]
            ),
            numeric_features,
        )
    ]
)

custom_scorer = make_scorer(f1_score, average="macro", greater_is_better=True)

class_weights = {0: 1, 1: 2, 2: 2, 3: 4, 4: 2}

models = {
    "Random Forest": {
        "pipeline": ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("sampling", SMOTEENN(random_state=SEED)),
                (
                    "model",
                    RandomForestClassifier(random_state=SEED, class_weight="balanced"),
                ),
            ]
        ),
        "params": {
            "model__n_estimators": [100, 500, 1000],
            "model__max_depth": [None, 15, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2"],
            "model__class_weight": [class_weights, "balanced"],
        },
    },
    "MLP Classifier": {
        "pipeline": ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("sampling", SMOTEENN(random_state=SEED)),
                ("model", MLPClassifier(random_state=SEED, max_iter=3000)),
            ]
        ),
        "params": {
            "model__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "model__activation": ["relu", "tanh"],
            "model__solver": ["adam", "sgd"],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.001, 0.01],
        },
    },
}


def evaluate_models(X, y):
    best_model = None
    best_score = 0

    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, config in models.items():
        print(f"\n{'='*30} Treinando {name} {'='*30}")

        search = RandomizedSearchCV(
            estimator=config["pipeline"],
            param_distributions=config["params"],
            n_iter=108,
            cv=cv_outer,
            scoring=custom_scorer,
            n_jobs=-1,
            random_state=SEED,
        )

        search.fit(X, y)

        cv_results = cross_validate(
            search.best_estimator_,
            X,
            y,
            cv=cv_outer,
            scoring={"f1": custom_scorer, "accuracy": "accuracy"},
            n_jobs=-1,
        )
        avg_f1 = np.mean(cv_results["test_f1"])
        avg_acc = np.mean(cv_results["test_accuracy"])
        print(f"F1 Macro: {avg_f1:.4f}")
        print(f"Acurácia: {avg_acc:.4f}")

        if avg_f1 > best_score:
            best_score = avg_f1
            best_model = search.best_estimator_

    return best_model


if __name__ == "__main__":
    X = train_df.drop(columns=["id", "target"])
    y = train_df["target"]

    print("\nIniciando otimização dos modelos...")
    final_model = evaluate_models(X, y)

    print("\nTreinando modelo final com todo o conjunto de treino...")
    final_model.fit(X, y)

    X_test = test_df.drop(columns=["id"])
    predictions = final_model.predict(X_test)

    submission = pd.DataFrame({"id": test_df["id"], "target": predictions})
    submission.to_csv("submission.csv", index=False)
    print("\nArquivo submission.csv gerado com sucesso!")

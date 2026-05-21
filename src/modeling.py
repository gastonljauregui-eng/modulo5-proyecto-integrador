from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def build_model(model_name, preprocessor):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(random_state=42)
    }

    if model_name not in models:
        raise ValueError(f"Modelo no reconocido: {model_name}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", models[model_name])
    ])

    return pipeline
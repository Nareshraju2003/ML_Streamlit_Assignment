import json
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Dict, Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


@dataclass
class ModelMetrics:
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    mcc: float


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def evaluate_binary_classifier(model, X_test, y_test) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    # probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: decision_function -> sigmoid-ish normalization
        scores = model.decision_function(X_test)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    metrics = ModelMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        auc=float(roc_auc_score(y_test, y_prob)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_test, y_pred)),
    )

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {"metrics": asdict(metrics), "confusion_matrix": cm, "report": report}


def main() -> None:
    # Dataset: UCI Breast Cancer Wisconsin (Diagnostic) via sklearn loader
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Define models (all on same dataset)
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=42)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=42
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    # Train & evaluate
    all_metrics: Dict[str, Any] = {}
    all_cm: Dict[str, Any] = {}

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        results = evaluate_binary_classifier(model, X_test, y_test)
        all_metrics[name] = results["metrics"]
        all_cm[name] = {
            "confusion_matrix": results["confusion_matrix"],
            "classification_report": results["report"],
        }

    # Save artifacts
    out_dir = os.path.join(os.path.dirname(__file__), ".")
    ensure_dir(out_dir)

    with open(os.path.join(out_dir, "models.pkl"), "wb") as f:
        pickle.dump(trained_models, f)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    with open(os.path.join(out_dir, "confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(all_cm, f, indent=2)

    with open(os.path.join(out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print("✅ Training complete.")
    print("Saved: model/models.pkl, model/metrics.json, model/confusion_matrix.json, model/feature_names.json")


if __name__ == "__main__":
    main()

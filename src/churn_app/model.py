"""Lightweight churn model: training, threshold tuning and evaluation.

The competition metric is balanced accuracy, so the decision threshold is
tuned for balanced accuracy on the validation split rather than left at 0.5.
The research code in this repository (``final_experiments/``) holds the
heavier Transformer / XGBoost models; this module is the fast, CPU-only
counterpart suited for interactive use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ModelType = Literal["gradient_boosting", "logistic_regression"]
MODEL_LABELS: dict[ModelType, str] = {
    "gradient_boosting": "Gradient boosting (HistGradientBoosting)",
    "logistic_regression": "Logistic regression",
}


@dataclass
class TrainResult:
    model: Pipeline
    threshold: float
    metrics: dict[str, float]
    confusion: pd.DataFrame
    importance: pd.DataFrame
    validation: pd.DataFrame  # userId-indexed: churn, proba, pred
    feature_names: list[str]


def make_model(model_type: ModelType, seed: int = 42) -> Pipeline:
    estimator: ClassifierMixin
    if model_type == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=seed,
        )
        return make_pipeline(estimator)
    if model_type == "logistic_regression":
        estimator = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
    raise ValueError(f"Unknown model_type {model_type!r}")


def best_threshold(y_true: np.ndarray, proba: np.ndarray, grid: np.ndarray | None = None) -> tuple[float, float]:
    """Threshold maximising balanced accuracy. Returns ``(threshold, score)``.

    Ties are broken towards the threshold closest to 0.5 for stability.
    """
    if grid is None:
        grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    scores = np.array([balanced_accuracy_score(y_true, (proba >= t).astype(int)) for t in grid])
    best = np.flatnonzero(np.isclose(scores, scores.max()))
    idx = best[np.argmin(np.abs(grid[best] - 0.5))]
    return float(grid[idx]), float(scores[idx])


def train_churn_model(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model_type: ModelType = "gradient_boosting",
    val_size: float = 0.25,
    seed: int = 42,
    importance_repeats: int = 5,
) -> TrainResult:
    """Fit a model on user features, tune the threshold and evaluate on a
    stratified hold-out split of users."""
    data = features.join(labels.set_index("userId")["churn"], how="inner")
    if data.empty:
        raise ValueError("No overlap between features and labels")
    y = data.pop("churn").astype(int).to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 4:
        raise ValueError("Need at least 4 churned and 4 retained users to train a model")

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=val_size, stratify=y, random_state=seed)
    model = make_model(model_type, seed)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)[:, 1]
    threshold, bal_acc = best_threshold(y_val, proba)
    pred = (proba >= threshold).astype(int)

    metrics = {
        "balanced_accuracy": bal_acc,
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "flagged_share": float(pred.mean()),
        "val_churn_rate": float(y_val.mean()),
        "n_train": float(len(y_train)),
        "n_val": float(len(y_val)),
    }
    cm = confusion_matrix(y_val, pred, labels=[0, 1])
    confusion = pd.DataFrame(
        cm, index=["actual: retained", "actual: churned"], columns=["predicted: retained", "predicted: churned"]
    )

    imp = permutation_importance(
        model, X_val, y_val, scoring="roc_auc", n_repeats=importance_repeats, random_state=seed
    )
    importance = (
        pd.DataFrame({"feature": data.columns, "importance": imp.importances_mean, "std": imp.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    validation = pd.DataFrame({"churn": y_val, "proba": proba, "pred": pred}, index=X_val.index)

    return TrainResult(
        model=model,
        threshold=threshold,
        metrics=metrics,
        confusion=confusion,
        importance=importance,
        validation=validation,
        feature_names=list(data.columns),
    )


def score_users(result: TrainResult, features: pd.DataFrame) -> pd.DataFrame:
    """Churn probability and 0/1 prediction for every user in ``features``."""
    X = features.reindex(columns=result.feature_names)
    proba = result.model.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {"churn_probability": proba, "predicted_churn": (proba >= result.threshold).astype(int)},
        index=features.index,
    ).sort_values("churn_probability", ascending=False)

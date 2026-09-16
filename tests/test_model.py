from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_app.data import churn_labels
from churn_app.features import build_user_features
from churn_app.model import best_threshold, make_model, score_users, train_churn_model


@pytest.fixture(scope="module")
def horizon_data(sample_events):
    cutoff = pd.Timestamp("2018-11-01")
    return build_user_features(sample_events, cutoff), churn_labels(sample_events, cutoff, 10)


def test_best_threshold_separable():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    threshold, score = best_threshold(y, proba)
    assert score == 1.0
    assert 0.3 < threshold <= 0.7
    assert threshold == pytest.approx(0.5)  # ties broken towards 0.5


def test_best_threshold_custom_grid():
    y = np.array([0, 1, 1, 0])
    proba = np.array([0.2, 0.8, 0.6, 0.4])
    assert best_threshold(y, proba, grid=np.array([0.5, 0.9])) == (0.5, 1.0)


@pytest.mark.parametrize("model_type", ["gradient_boosting", "logistic_regression"])
def test_train_churn_model(horizon_data, model_type):
    features, labels = horizon_data
    result = train_churn_model(features, labels, model_type, importance_repeats=2)
    m = result.metrics
    assert 0.5 <= m["balanced_accuracy"] <= 1
    assert 0.5 <= m["roc_auc"] <= 1
    assert m["n_train"] + m["n_val"] == len(features.join(labels.set_index("userId"), how="inner"))
    assert result.confusion.to_numpy().sum() == m["n_val"]
    assert set(result.importance["feature"]) == set(features.columns)
    assert result.validation["pred"].isin([0, 1]).all()


def test_training_is_reproducible(horizon_data):
    features, labels = horizon_data
    a = train_churn_model(features, labels, seed=3, importance_repeats=2)
    b = train_churn_model(features, labels, seed=3, importance_repeats=2)
    assert a.metrics == b.metrics
    pd.testing.assert_frame_equal(a.validation, b.validation)


def test_score_users_sorted_and_aligned(horizon_data):
    features, labels = horizon_data
    result = train_churn_model(features, labels, "logistic_regression", importance_repeats=1)
    scores = score_users(result, features)
    assert set(scores.index) == set(features.index)
    assert scores["churn_probability"].is_monotonic_decreasing
    assert (scores["predicted_churn"] == (scores["churn_probability"] >= result.threshold)).all()


def test_train_rejects_single_class(horizon_data):
    features, labels = horizon_data
    with pytest.raises(ValueError, match="at least 4"):
        train_churn_model(features, labels.assign(churn=0))


def test_train_rejects_no_overlap(horizon_data):
    features, _ = horizon_data
    labels = pd.DataFrame({"userId": ["nobody"], "churn": [1]})
    with pytest.raises(ValueError, match="No overlap"):
        train_churn_model(features, labels)


def test_unknown_model_type():
    with pytest.raises(ValueError, match="Unknown model_type"):
        make_model("svm")  # type: ignore[arg-type]

"""
Training and inference script for an XGBoost baseline (standalone version).
"""
#%%
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from .xgb_user_features import prepare_user_level_datasets

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "churn-prediction-25-26/train.parquet"
DEFAULT_TEST_PATH = PROJECT_ROOT / "churn-prediction-25-26/test.parquet"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"
DEFAULT_SUBMISSION_PATH = DEFAULT_OUTPUT_DIR / "xgb_submission.csv"


def train_xgb_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_train: Sequence[int],
    y_val: Sequence[int],
    feature_cols: Sequence[str],
) -> XGBClassifier:
    """
    Fit an XGBoost classifier with reasonable defaults.
    """
    pos_count = float(np.sum(y_train))
    neg_count = float(len(y_train) - pos_count)
    scale_pos_weight = neg_count / max(pos_count, 1.0)

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=8,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )

    fit_kwargs = {}
    if len(val_df) > 0:
        fit_kwargs["eval_set"] = [(val_df[feature_cols], y_val)]
        fit_kwargs["verbose"] = False

    model.fit(train_df[feature_cols], y_train, **fit_kwargs)
    return model


def run_pipeline(
    train_path: Path | str = DEFAULT_TRAIN_PATH,
    test_path: Path | str = DEFAULT_TEST_PATH,
    output_path: Path | str = DEFAULT_SUBMISSION_PATH,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[XGBClassifier, float | None, Path]:
    """
    End-to-end training + prediction helper.
    """
    (
        train_df,
        val_df,
        test_df,
        y_train,
        y_val,
        artifacts,
    ) = prepare_user_level_datasets(
        train_path=train_path,
        test_path=test_path,
        val_ratio=val_ratio,
        random_state=random_state,
    )

    feature_cols = artifacts.feature_names
    model = train_xgb_classifier(train_df, val_df, y_train, y_val, feature_cols)

    val_logloss = None
    if len(val_df) > 0:
        val_pred = model.predict_proba(val_df[feature_cols])[:, 1]
        val_logloss = log_loss(y_val, val_pred)

    test_pred = model.predict_proba(test_df[feature_cols])[:, 1]
    submission = pd.DataFrame({"id": test_df["userId"], "target": test_pred})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    return model, val_logloss, output_path


if __name__ == "__main__":
    model, val_logloss, saved_path = run_pipeline()
    if val_logloss is not None:
        print(f"Validation logloss: {val_logloss:.5f}")
    print(f"Saved predictions to {saved_path}")

# %%

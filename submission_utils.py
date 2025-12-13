from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import clone


def _predict_proba_1d(model, X) -> np.ndarray:
    """Return positive-class probabilities, clipping regression outputs if needed."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.clip(model.predict(X), 0, 1)


def fit_and_save_submission(
    model,
    X_train,
    y_train,
    X_test,
    out_path: Union[str, Path],
    threshold: float = 0.5,
):
    """Fit a clone of the estimator, write submission CSV, and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    estimator = clone(model)
    estimator.fit(X_train, y_train)

    test_proba = _predict_proba_1d(estimator, X_test)
    submission = pd.DataFrame(
        {"id": X_test.index.astype(str), "target": (test_proba >= threshold).astype(int)}
    )
    submission.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return out_path


def fit_and_save_many(
    models: Dict[str, object],
    X_train,
    y_train,
    X_test,
    out_dir: Union[str, Path],
    threshold: float = 0.5,
) -> Sequence[Path]:
    """Fit multiple estimators and save a submission file for each."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for name, model in models.items():
        out_path = out_dir / f"submission_{name}.csv"
        saved_paths.append(
            fit_and_save_submission(
                model, X_train, y_train, X_test, out_path, threshold=threshold
            )
        )
    return saved_paths


# Aliases to satisfy both naming conventions used in notebooks
fit_save_submission = fit_and_save_submission
fit_save_many = fit_and_save_many

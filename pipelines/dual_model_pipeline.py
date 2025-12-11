"""
Unified pipelines for training/inference of Transformer and XGBoost models.

Everything is isolated under `pipelines/` so the original scripts remain
unchanged. Each helper returns paths/metrics so notebooks can pick the pieces
they need without duplicating logic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import torch
from torch.utils.data import DataLoader

from pipelines.data_bundles import (
    TransformerBundle,
    XGBBundle,
    prepare_transformer_bundle,
    prepare_xgb_bundle,
)
from transformer_model import (
    TrainingConfig,
    ChurnTransformer,
    UserSequenceDataset,
    collate_batch,
    make_dataloaders,
    predict_proba,
    train_model,
)
from xgboost_model.xgb_train import train_xgb_classifier

PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "churn-prediction-25-26"
DEFAULT_TRAIN_PATH = DEFAULT_DATA_DIR / "train.parquet"
DEFAULT_TEST_PATH = DEFAULT_DATA_DIR / "test.parquet"
OUTPUT_ROOT = PIPELINE_ROOT / "outputs"
TRANSFORMER_OUTPUT_DIR = OUTPUT_ROOT / "transformer"
XGB_OUTPUT_DIR = OUTPUT_ROOT / "xgboost"


def _compute_pos_weight(labels: Dict[object, int], user_ids: Optional[Iterable[object]] = None) -> float:
    """
    Compute the pos_weight term for BCE loss based on provided user IDs.
    """
    if user_ids is None:
        relevant = labels.values()
    else:
        relevant = (labels[uid] for uid in user_ids)
    labels_arr = np.fromiter(relevant, dtype=float)
    pos = labels_arr.sum()
    neg = float(len(labels_arr) - pos)
    return neg / max(pos, 1.0)


def _make_prediction_loader(
    sequences: Dict[str, "SequenceData"],
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader for inference (labels are dummy placeholders).
    """
    dataset = UserSequenceDataset(sequences, labels=None, user_ids=list(sequences.keys()))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )


def run_transformer_pipeline(
    bundle: Optional[TransformerBundle] = None,
    *,
    train_path: Union[str, Path] = DEFAULT_TRAIN_PATH,
    test_path: Union[str, Path] = DEFAULT_TEST_PATH,
    config: Optional[TrainingConfig] = None,
    output_dir: Union[str, Path] = TRANSFORMER_OUTPUT_DIR,
    **prep_kwargs,
) -> Dict[str, object]:
    """
    Train the Transformer model and export predictions/metrics.
    - bundle: optionally pass a precomputed TransformerBundle to skip feature prep.
    - prep_kwargs are forwarded to prepare_transformer_bundle (cutoff, cache_dir, etc.).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = TrainingConfig()

    if bundle is None:
        prep_kwargs = dict(prep_kwargs)
        prep_kwargs.setdefault("max_seq_len", config.max_seq_len)
        bundle = prepare_transformer_bundle(
            train_path=train_path,
            test_path=test_path,
            **prep_kwargs,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = bundle.labels
    train_loader, val_loader = make_dataloaders(
        bundle.train_sequences, bundle.val_sequences, labels, config
    )

    artifacts = bundle.artifacts
    model = ChurnTransformer(
        num_numeric=len(artifacts.numeric_cols),
        num_pages=len(artifacts.page_categories),
        num_metro=len(artifacts.metro_mapping),
        num_state=len(artifacts.state_mapping),
        num_device=len(artifacts.device_mapping),
        config=config,
    ).to(device)

    pos_weight = _compute_pos_weight(labels, bundle.train_sequences.keys())
    train_history, val_history, model, best_epoch, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        pos_weight=pos_weight,
        use_cosine_decay=config.use_cosine_decay,
        eta_min_factor=config.eta_min_factor,
        warmup_epochs=config.warmup_epochs,
        use_focal_loss=config.use_focal_loss,
        focal_gamma=config.focal_gamma,
        early_stop_patience=config.early_stop_patience,
        early_stop_min_delta=config.early_stop_min_delta,
    )

    # Validation metric
    val_probs = predict_proba(model, val_loader, device)
    val_user_ids = list(val_probs.keys())
    val_logloss = None
    if len(val_user_ids) > 0:
        val_labels = bundle.label_series.loc[val_user_ids]
        val_preds = [val_probs[uid] for uid in val_user_ids]
        val_logloss = log_loss(val_labels, val_preds)

    # Test predictions
    test_loader = _make_prediction_loader(
        bundle.test_sequences, batch_size=config.batch_size, num_workers=config.num_workers
    )
    test_probs = predict_proba(model, test_loader, device)
    submission = pd.DataFrame(
        {
            "id": list(test_probs.keys()),
            "target": [int(p >= 0.5) for p in test_probs.values()],
            "probability": list(test_probs.values()),
        }
    )
    submission_path = output_dir / "transformer_submission.csv"
    submission.to_csv(submission_path, index=False)

    model_path = output_dir / "transformer_model.pt"
    torch.save(best_state, model_path)

    metrics = {
        "val_logloss": val_logloss,
        "best_epoch": best_epoch,
        "train_history": train_history,
        "val_history": val_history,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)

    return {
        "model_path": model_path,
        "submission_path": submission_path,
        "metrics_path": metrics_path,
        "config_path": config_path,
        "val_logloss": val_logloss,
    }


def run_xgb_pipeline(
    bundle: Optional[XGBBundle] = None,
    *,
    train_path: Union[str, Path] = DEFAULT_TRAIN_PATH,
    test_path: Union[str, Path] = DEFAULT_TEST_PATH,
    output_dir: Union[str, Path] = XGB_OUTPUT_DIR,
    **prep_kwargs,
) -> Dict[str, object]:
    """
    Train the XGBoost baseline and export predictions/metrics.
    - bundle: optionally pass a precomputed XGBBundle to skip feature prep.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if bundle is None:
        bundle = prepare_xgb_bundle(
            train_path=train_path,
            test_path=test_path,
            **prep_kwargs,
        )

    feature_cols = bundle.artifacts.feature_names
    model = train_xgb_classifier(
        train_df=bundle.train_df,
        val_df=bundle.val_df,
        y_train=bundle.y_train,
        y_val=bundle.y_val,
        feature_cols=feature_cols,
    )

    val_logloss = None
    if len(bundle.val_df) > 0:
        val_pred = model.predict_proba(bundle.val_df[feature_cols])[:, 1]
        val_logloss = log_loss(bundle.y_val, val_pred)

    test_pred = model.predict_proba(bundle.test_df[feature_cols])[:, 1]
    submission = pd.DataFrame(
        {
            "id": bundle.test_df["userId"],
            "target": (test_pred >= 0.5).astype(int),
            "probability": test_pred,
        }
    )
    submission_path = output_dir / "xgb_submission.csv"
    submission.to_csv(submission_path, index=False)

    model_path = output_dir / "xgb_model.json"
    model.save_model(model_path)

    metrics = {"val_logloss": val_logloss}
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": model_path,
        "submission_path": submission_path,
        "metrics_path": metrics_path,
        "val_logloss": val_logloss,
    }


def run_full_pipeline(
    *,
    train_path: Union[str, Path] = DEFAULT_TRAIN_PATH,
    test_path: Union[str, Path] = DEFAULT_TEST_PATH,
    transformer_config: Optional[TrainingConfig] = None,
    transformer_prep_kwargs: Optional[Dict[str, object]] = None,
    xgb_prep_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Convenience wrapper to run both models back-to-back.
    """
    transformer_result = run_transformer_pipeline(
        train_path=train_path,
        test_path=test_path,
        config=transformer_config,
        **(transformer_prep_kwargs or {}),
    )
    xgb_result = run_xgb_pipeline(
        train_path=train_path,
        test_path=test_path,
        **(xgb_prep_kwargs or {}),
    )

    summary = {"transformer": transformer_result, "xgboost": xgb_result}
    summary_path = OUTPUT_ROOT / "pipeline_summary.json"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    print("Use run_full_pipeline(...) or the individual helpers from notebooks/scripts.")

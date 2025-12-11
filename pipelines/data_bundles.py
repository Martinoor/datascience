"""
Data preparation helpers for the combined Transformer + XGBoost pipelines.

These wrappers keep the original modules untouched while exposing
ready-to-train bundles that live entirely inside `pipelines/`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from feature_pipeline import FeatureArtifacts, prepare_datasets
from transformer_model import SequenceData, build_user_sequences
from xgboost_model.xgb_user_features import (
    XGBFeatureArtifacts,
    prepare_user_level_datasets,
)

# Normalize the error column name so downstream code does not have to deal with spaces.
ERROR_RENAME_MAP = {"error occur": "error_occur"}


@dataclass
class TransformerBundle:
    train_sequences: Dict[str, SequenceData]
    val_sequences: Dict[str, SequenceData]
    test_sequences: Dict[str, SequenceData]
    labels: Dict[str, int]
    label_series: pd.Series
    artifacts: FeatureArtifacts


@dataclass
class XGBBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    artifacts: XGBFeatureArtifacts


def _rename_error_feature(df: pd.DataFrame, artifacts: FeatureArtifacts):
    """
    Rename the error column to an underscore version and update numeric cols metadata.
    """
    df = df.rename(columns=ERROR_RENAME_MAP)
    numeric_cols = [ERROR_RENAME_MAP.get(c, c) for c in artifacts.numeric_cols]
    artifacts = replace(artifacts, numeric_cols=numeric_cols)
    return df, artifacts


def prepare_transformer_bundle(
    train_path: Union[str, Path],
    test_path: Union[str, Path],
    *,
    val_ratio: float = 0.2,
    random_state: int = 42,
    max_seq_len: int = 400,
    cutoff_time: Optional[Union[pd.Timestamp, str, int]] = None,
    drop_inactive_before_cutoff: bool = False,
    truncate_buffer_min: int = 2,
    truncate_buffer_frac: float = 0.1,
    cache_dir: Union[str, Path] = "feature_cache_v2",
    use_cache: bool = True,
    rename_error_feature: bool = True,
) -> TransformerBundle:
    """
    Build per-user sequences ready for Transformer training/validation/testing.
    """
    train_df, val_df, test_df, labels, artifacts = prepare_datasets(
        train_path=train_path,
        test_path=test_path,
        val_ratio=val_ratio,
        random_state=random_state,
        truncate_buffer_min=truncate_buffer_min,
        truncate_buffer_frac=truncate_buffer_frac,
        cutoff_time=cutoff_time,
        drop_inactive_before_cutoff=drop_inactive_before_cutoff,
        use_cache=use_cache,
        cache_dir=str(cache_dir),
    )

    if rename_error_feature:
        train_df, artifacts = _rename_error_feature(train_df, artifacts)
        val_df = val_df.rename(columns=ERROR_RENAME_MAP)
        test_df = test_df.rename(columns=ERROR_RENAME_MAP)

    numeric_cols = artifacts.numeric_cols
    train_seqs = build_user_sequences(train_df, numeric_cols, max_seq_len)
    val_seqs = build_user_sequences(val_df, numeric_cols, max_seq_len)
    test_seqs = build_user_sequences(test_df, numeric_cols, max_seq_len)

    label_dict = {uid: int(label) for uid, label in labels.items()}
    return TransformerBundle(
        train_sequences=train_seqs,
        val_sequences=val_seqs,
        test_sequences=test_seqs,
        labels=label_dict,
        label_series=labels,
        artifacts=artifacts,
    )


def prepare_xgb_bundle(
    train_path: Union[str, Path],
    test_path: Union[str, Path],
    *,
    val_ratio: float = 0.2,
    random_state: int = 42,
    cutoff_time: Optional[Union[pd.Timestamp, str, int]] = "2018-11-10",
    drop_inactive_before_cutoff: bool = False,
    truncate_buffer_min: int = 3,
    truncate_buffer_frac: float = 0.2,
) -> XGBBundle:
    """
    Build user-level aggregated features for the XGBoost baseline.
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
        truncate_buffer_min=truncate_buffer_min,
        truncate_buffer_frac=truncate_buffer_frac,
        cutoff_time=cutoff_time,
        drop_inactive_before_cutoff=drop_inactive_before_cutoff,
    )

    return XGBBundle(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        y_train=y_train,
        y_val=y_val,
        artifacts=artifacts,
    )

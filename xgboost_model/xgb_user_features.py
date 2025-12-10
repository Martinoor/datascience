"""
Standalone user-level feature helpers for the XGBoost baseline.

Lives inside the `xgboost_model` package and reuses the existing
feature engineering utilities in `feature_pipeline.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from feature_pipeline import (
    _build_mapping,
    encode_categoricals,
    feature_engineer,
    truncate_last_days_per_user,
    truncate_user_histories,
)


@dataclass
class XGBFeatureArtifacts:
    """Metadata produced during the XGBoost feature preparation."""

    page_categories: List[str]
    metro_mapping: Dict[str, int]
    state_mapping: Dict[str, int]
    device_mapping: Dict[str, int]
    base_numeric_cols: List[str]
    feature_names: List[str]
    scaler: StandardScaler


def _align_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure train/test have identical columns (useful for page dummies that
    might only appear in one split).
    """
    all_cols = sorted(set(train_df.columns) | set(test_df.columns))
    for col in all_cols:
        if col not in train_df:
            train_df[col] = 0
        if col not in test_df:
            test_df[col] = 0
    train_df = train_df[all_cols]
    test_df = test_df[all_cols]
    return train_df, test_df


def _collect_numeric_columns(df: pd.DataFrame, cat_id_cols: Sequence[str]) -> List[str]:
    """
    Pick numeric-like columns to aggregate (drop ids handled separately).
    """
    numeric_cols: List[str] = []
    for col in df.columns:
        if col in {"userId", "time"} or col in cat_id_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return sorted(numeric_cols)


def aggregate_user_features(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    cat_id_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Aggregate event-level rows into a single user-level feature vector.
    """
    df_sorted = df.sort_values(["userId", "time"])
    rows: List[Dict[str, float]] = []

    for user_id, g in df_sorted.groupby("userId", sort=False):
        stats: Dict[str, float] = {"userId": user_id}
        stats["event_count"] = float(len(g))

        span_seconds = (g["time"].iloc[-1] - g["time"].iloc[0]).total_seconds() if len(g) > 1 else 0.0
        stats["active_hours"] = span_seconds / 3600.0
        stats["events_per_hour"] = stats["event_count"] / max(stats["active_hours"], 1e-3)

        gap_seconds = g["time"].diff().dt.total_seconds().dropna()
        stats["median_gap_seconds"] = float(gap_seconds.median()) if not gap_seconds.empty else 0.0
        stats["mean_gap_seconds"] = float(gap_seconds.mean()) if not gap_seconds.empty else 0.0

        stats["days_since_registration_last"] = g["regis_time_seconds"].iloc[-1] / 86400.0
        stats["paid_ratio"] = g["level"].mean()
        stats["paid_last"] = float(g["level"].iloc[-1])
        stats["level_switches"] = float(g["level"].diff().fillna(0).abs().sum())

        # Categorical IDs: keep last state and diversity counts.
        for col in cat_id_cols:
            stats[f"{col}_last"] = float(g[col].iloc[-1])
            stats[f"{col}_nunique"] = float(g[col].nunique())

        for col in numeric_cols:
            series = g[col]
            stats[f"{col}_mean"] = float(series.mean())
            stats[f"{col}_std"] = float(series.std(ddof=0))
            stats[f"{col}_max"] = float(series.max())
            stats[f"{col}_last"] = float(series.iloc[-1])

        rows.append(stats)

    user_df = pd.DataFrame(rows)
    user_df = user_df.fillna(0.0)
    return user_df


def prepare_user_level_datasets(
    train_path: Path | str,
    test_path: Path | str,
    val_ratio: float = 0.2,
    random_state: int = 42,
    truncate_buffer_min: int = 3,
    truncate_buffer_frac: float = 0.2,
    cutoff_time: Optional[Union[pd.Timestamp, str, int]] = None,
    drop_inactive_before_cutoff: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, XGBFeatureArtifacts]:
    """
    Build user-level matrices for XGBoost training and inference.
    Labels always reflect the full sequence (normal marking), while training
    features can be restricted by a cutoff time to avoid leakage from later
    behaviors.
    """
    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)

    labels = (
        train_raw.groupby("userId")["page"]
        .apply(lambda s: int((s == "Cancellation Confirmation").any()))
        .astype(int)
    )

    # Optionally drop training events after a cutoff (keep labels untouched).
    if cutoff_time is not None:
        if isinstance(cutoff_time, Integral):
            train_raw = truncate_last_days_per_user(train_raw, int(cutoff_time))
            if drop_inactive_before_cutoff:
                active_users = train_raw["userId"].unique()
                labels = labels[labels.index.isin(active_users)]
        else:
            cutoff_ts = pd.to_datetime(cutoff_time)
            train_raw = train_raw[train_raw["time"] <= cutoff_ts].copy()
            if drop_inactive_before_cutoff:
                active_users = train_raw["userId"].unique()
                labels = labels[labels.index.isin(active_users)]

    # Remove explicit cancellation rows to avoid leakage.
    train_raw = train_raw[train_raw["page"] != "Cancellation Confirmation"].copy()

    train_raw = truncate_user_histories(
        train_raw, buffer_min=truncate_buffer_min, buffer_frac=truncate_buffer_frac
    )

    page_categories = sorted(set(train_raw["page"]).union(set(test_raw["page"])))
    train_fe = feature_engineer(train_raw, page_categories)
    test_fe = feature_engineer(test_raw, page_categories)

    metro_map = _build_mapping([train_fe["metro"], test_fe["metro"]])
    state_map = _build_mapping([train_fe["state"], test_fe["state"]])
    device_map = _build_mapping([train_fe["device"], test_fe["device"]])

    train_fe = encode_categoricals(train_fe, page_categories, metro_map, state_map, device_map)
    test_fe = encode_categoricals(test_fe, page_categories, metro_map, state_map, device_map)

    # Renaming avoids spaces in column names for downstream models.
    for df in (train_fe, test_fe):
        if "error occur" in df.columns:
            df.rename(columns={"error occur": "error_occur"}, inplace=True)
        df["regis_time_seconds"] = df["regis_time"].dt.total_seconds()

    drop_object_cols = ["regis_time", "song", "artist", "page", "metro", "state", "device"]
    train_fe = train_fe.drop(columns=[c for c in drop_object_cols if c in train_fe.columns])
    test_fe = test_fe.drop(columns=[c for c in drop_object_cols if c in test_fe.columns])

    train_fe, test_fe = _align_columns(train_fe, test_fe)

    cat_id_cols = ["page_id", "metro_id", "state_id", "device_id"]
    numeric_cols = _collect_numeric_columns(train_fe, cat_id_cols)

    train_user = aggregate_user_features(train_fe, numeric_cols, cat_id_cols)
    test_user = aggregate_user_features(test_fe, numeric_cols, cat_id_cols)

    stratify = labels if labels.value_counts().min() >= 2 else None
    train_users, val_users = train_test_split(
        labels.index,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify,
    )

    train_user_df = train_user[train_user["userId"].isin(set(train_users))].reset_index(drop=True)
    val_user_df = train_user[train_user["userId"].isin(set(val_users))].reset_index(drop=True)

    feature_cols = [c for c in train_user_df.columns if c != "userId"]
    scaler = StandardScaler()
    train_user_df[feature_cols] = scaler.fit_transform(train_user_df[feature_cols])
    val_user_df[feature_cols] = scaler.transform(val_user_df[feature_cols])
    test_user[feature_cols] = scaler.transform(test_user[feature_cols])

    artifacts = XGBFeatureArtifacts(
        page_categories=list(page_categories),
        metro_mapping=metro_map,
        state_mapping=state_map,
        device_mapping=device_map,
        base_numeric_cols=list(numeric_cols),
        feature_names=list(feature_cols),
        scaler=scaler,
    )

    y_train = pd.Series(labels.loc[train_user_df["userId"]].values, index=train_user_df["userId"])
    y_val = pd.Series(labels.loc[val_user_df["userId"]].values, index=val_user_df["userId"])

    return train_user_df, val_user_df, test_user, y_train, y_val, artifacts

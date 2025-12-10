"""
Reusable feature engineering pipeline for the churn dataset.

The logic mirrors the exploration done in `feature_engineering.ipynb` but is
packaged so it can be reused by training / inference scripts and applied to
both train and test consistently.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

FEATURE_PIPELINE_VERSION = "1.1"

# Key pages to track sequence statistics for (counts / rolling ratios).
KEY_PAGES: List[str] = [
    "Help",
    "Settings",
    "Upgrade",
    "Downgrade",
    "Submit Upgrade",
    "Submit Downgrade",
    "Logout",
    "Error",
    "Add Friend",
]
KEY_PAGE_ROLLING_WINDOW = 50


@dataclass
class FeatureArtifacts:
    """Metadata produced during feature processing."""

    page_categories: List[str]
    metro_mapping: Dict[str, int]
    state_mapping: Dict[str, int]
    device_mapping: Dict[str, int]
    numeric_cols: List[str]
    scaler: StandardScaler


def detect_device(user_agent: str) -> str:
    """Rudimentary UA parsing used in the original notebook."""
    if not isinstance(user_agent, str):
        return "Other"

    ua = user_agent.lower()
    if "iphone" in ua:
        return "iPhone"
    if "ipad" in ua:
        return "iPad"
    if "macintosh" in ua or "mac os x" in ua:
        return "Mac"
    if "windows" in ua:
        return "Windows"
    if "android" in ua:
        return "Android"
    if "linux" in ua or "ubuntu" in ua:
        return "Linux"
    return "Other"


def truncate_user_histories(
    df: pd.DataFrame,
    buffer_min: int = 2,
    buffer_frac: float = 0.1,
) -> pd.DataFrame:
    """
    Trim the latest part of each user's timeline to mimic test distribution and
    reduce leakage from behaviors right before churn events.
    """
    kept = []
    grouped = df.sort_values(["userId", "time"]).groupby("userId", sort=False)
    for _, g in tqdm(grouped, total=grouped.ngroups, desc="truncate_user_histories"):
        L = len(g)
        buffer = max(buffer_min, int(L * buffer_frac))
        if L <= 1:
            kept.append(g)
            continue
        cutoff = max(L - buffer, 1)
        kept.append(g.iloc[:cutoff])
    return pd.concat(kept, ignore_index=True)


def truncate_last_days_per_user(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Remove the most recent `days` worth of events for each user to simulate
    forecasting into a future window.
    """
    if days <= 0:
        return df.copy()
    delta = pd.Timedelta(days=days)
    user_max = df.groupby("userId")["time"].transform("max")
    cutoff = user_max - delta
    return df[df["time"] <= cutoff].copy()


def _build_mapping(series_list: Sequence[pd.Series]) -> Dict[str, int]:
    """Create a stable categorical mapping."""
    values: List[str] = []
    for s in series_list:
        values.extend(list(s.dropna().unique()))
    categories = sorted(set(values))
    return {cat: idx + 1 for idx, cat in enumerate(categories)}  # 0 = unknown/pad


def compute_error_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Running ratio of historical 404s per user.

    Mirrors the `cum_404_ratio` logic from the notebook but vectorized.
    """
    df_sorted = df.sort_values(["userId", "time"])
    is_404 = (df_sorted["status"] == 404).astype(int)
    cum_404 = is_404.groupby(df_sorted["userId"]).cumsum().shift(fill_value=0)
    counts = df_sorted.groupby("userId").cumcount() + 1
    ratio = (cum_404 / counts).astype(np.float32)
    return ratio.reindex(df.index)


def get_song_stats_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute incremental song/artist stats per user (same as the notebook).
    """
    df = df.sort_values(["userId", "time"]).reset_index(drop=True)
    df["_song_valid"] = df["song"].where(
        (~df["song"].isin(["None"])) & (df["song"].notna()), other=np.nan
    )
    df["_artist_valid"] = df["artist"].where(
        (~df["artist"].isin(["None"])) & (df["artist"].notna()), other=np.nan
    )

    n = df.shape[0]
    user_song_count = np.zeros(n, dtype=np.int32)
    user_artist_count = np.zeros(n, dtype=np.int32)

    song_top1_frac = np.full(n, np.nan, dtype=float)
    song_top1_count = np.zeros(n, dtype=np.int32)
    song_top3_frac = np.full(n, np.nan, dtype=float)
    song_top3_count = np.zeros(n, dtype=np.int32)

    artist_top1_frac = np.full(n, np.nan, dtype=float)
    artist_top1_count = np.zeros(n, dtype=np.int32)
    artist_top3_frac = np.full(n, np.nan, dtype=float)
    artist_top3_count = np.zeros(n, dtype=np.int32)

    last_user = None
    song_seen: set = set()
    artist_seen: set = set()
    song_counts: Dict[str, int] = {}
    artist_counts: Dict[str, int] = {}

    total_song_cnt = 0
    total_artist_cnt = 0
    song_top1 = 0
    song_top3 = [0, 0, 0]
    artist_top1 = 0
    artist_top3 = [0, 0, 0]

    def update_top3(top3_list: List[int], new_cnt: int) -> List[int]:
        c1, c2, c3 = top3_list
        if new_cnt <= c3:
            return [c1, c2, c3]
        if new_cnt >= c1:
            return [new_cnt, c1, c2]
        if new_cnt >= c2:
            return [c1, new_cnt, c2]
        return [c1, c2, new_cnt]

    for idx in tqdm(range(n)):
        uid = df.at[idx, "userId"]
        song = df.at[idx, "_song_valid"]
        artist = df.at[idx, "_artist_valid"]

        if uid != last_user:
            last_user = uid
            song_seen.clear()
            artist_seen.clear()
            song_counts.clear()
            artist_counts.clear()
            total_song_cnt = 0
            total_artist_cnt = 0
            song_top1 = 0
            song_top3 = [0, 0, 0]
            artist_top1 = 0
            artist_top3 = [0, 0, 0]

        if isinstance(song, str):
            song_seen.add(song)
            song_counts[song] = song_counts.get(song, 0) + 1
            total_song_cnt += 1
            new_cnt = song_counts[song]
            song_top1 = max(song_top1, new_cnt)
            song_top3 = update_top3(song_top3, new_cnt)

        if isinstance(artist, str):
            artist_seen.add(artist)
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            total_artist_cnt += 1
            new_cnt_a = artist_counts[artist]
            artist_top1 = max(artist_top1, new_cnt_a)
            artist_top3 = update_top3(artist_top3, new_cnt_a)

        user_song_count[idx] = len(song_seen)
        user_artist_count[idx] = len(artist_seen)

        if total_song_cnt > 0:
            song_top1_count[idx] = song_top1
            song_top3_count[idx] = sum(song_top3)
            song_top1_frac[idx] = song_top1 / total_song_cnt
            song_top3_frac[idx] = song_top3_count[idx] / total_song_cnt
        else:
            song_top1_count[idx] = 0
            song_top3_count[idx] = 0
            song_top1_frac[idx] = 0.0
            song_top3_frac[idx] = 0.0

        if total_artist_cnt > 0:
            artist_top1_count[idx] = artist_top1
            artist_top3_count[idx] = sum(artist_top3)
            artist_top1_frac[idx] = artist_top1 / total_artist_cnt
            artist_top3_frac[idx] = artist_top3_count[idx] / total_artist_cnt
        else:
            artist_top1_count[idx] = 0
            artist_top3_count[idx] = 0
            artist_top1_frac[idx] = 0.0
            artist_top3_frac[idx] = 0.0

    df["num_distinct_song_until_now"] = user_song_count  # Unique songs seen by the user so far
    df["num_distinct_artist_until_now"] = user_artist_count  # Unique artists seen so far
    df["song_top1_frac_until_now"] = song_top1_frac  # Share of plays made up by the user's top song
    df["song_top1_count_until_now"] = song_top1_count  # Play count of the most played song so far
    df["song_top3_frac_until_now"] = song_top3_frac  # Share of plays from the top 3 songs combined
    df["song_top3_count_until_now"] = song_top3_count  # Total plays across the top 3 songs
    df["artist_top1_frac_until_now"] = artist_top1_frac  # Share of plays from the top artist
    df["artist_top1_count_until_now"] = artist_top1_count  # Play count of the top artist
    df["artist_top3_frac_until_now"] = artist_top3_frac  # Share of plays from the top 3 artists
    df["artist_top3_count_until_now"] = artist_top3_count  # Total plays across the top 3 artists

    df = df.drop(columns=["_song_valid", "_artist_valid"])
    return df


def _add_page_dummies(df: pd.DataFrame, page_categories: Sequence[str]) -> pd.DataFrame:
    cat_type = pd.api.types.CategoricalDtype(categories=page_categories)
    df["page"] = df["page"].astype(cat_type)
    dummies = pd.get_dummies(df["page"])
    df = pd.concat([df, dummies], axis=1)
    return df


def _safe_page_feature_name(page: str) -> str:
    """Turn a raw page name into a safe, concise column name fragment."""
    name = page.strip().lower()
    for ch in [" ", "/", "-", "(", ")", "[", "]"]:
        name = name.replace(ch, "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def _add_page_sequence_features(df: pd.DataFrame, window: int = KEY_PAGE_ROLLING_WINDOW) -> pd.DataFrame:
    """
    For each key page, add:
    - cumulative count up to current event per user
    - rolling ratio over the last `window` events per user

    Implemented with groupby + cumsum / rolling to stay vectorized and efficient.
    """
    if "userId" not in df or "page" not in df:
        return df

    user_ids = df["userId"]
    for page in KEY_PAGES:
        safe = _safe_page_feature_name(page)
        flag = (df["page"] == page).astype(np.int8)

        count_col = f"{safe}_count_until_now"
        ratio_col = f"{safe}_ratio_last_{window}_events"

        # Cumulative count of this page for each user.
        df[count_col] = flag.groupby(user_ids).cumsum().astype(np.int32)

        # Rolling ratio of this page in the last `window` events for each user.
        rolling = (
            flag.groupby(user_ids)
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[ratio_col] = rolling.astype(np.float32)

    return df


def feature_engineer(df: pd.DataFrame, page_categories: Sequence[str]) -> pd.DataFrame:
    """
    Apply the feature engineering steps to a raw dataframe.
    """
    df = df.copy()
    to_drop = ["firstName", "lastName", "ts", "auth", "itemInSession", "sessionId", "method"]
    df = df.drop(columns=to_drop)
    df = df.sort_values(["userId", "time"])

    df["error occur"] = compute_error_ratio(df)  # Cumulative ratio of 404 responses up to each event
    df = df.drop(columns=["status"])

    df["gender"] = df["gender"].map({"F": 0, "M": 1}).fillna(0).astype(np.int64)  # Binary gender flag
    df["level"] = df["level"].map({"free": 0, "paid": 1}).fillna(0).astype(np.int64)  # Paid vs free level

    df = _add_page_dummies(df, page_categories)  # One-hot indicators for every page category

    # Add sequence-based features for a handful of key pages (Help, Settings, etc.).
    # This operates only on non-cancellation pages (train_raw has had cancellation rows removed),
    # so we avoid direct label leakage.
    df = _add_page_sequence_features(df)

    df[["metro", "state"]] = df["location"].str.rsplit(", ", n=1, expand=True)  # Split city/state from location
    df = df.drop(columns=["location"])

    df["device"] = df["userAgent"].apply(detect_device)  # Parsed device type from user agent
    df = df.drop(columns=["userAgent"])

    df = get_song_stats_fast(df)  # Rolling song/artist diversity and concentration stats
    df["regis_time"] = df.time - df.registration  # Time since registration for each event
    df = df.drop(columns=["registration"])
    return df


def encode_categoricals(
    df: pd.DataFrame,
    page_categories: Sequence[str],
    metro_mapping: Dict[str, int],
    state_mapping: Dict[str, int],
    device_mapping: Dict[str, int],
) -> pd.DataFrame:
    page_idx = {cat: idx + 1 for idx, cat in enumerate(page_categories)}  # 0 = pad
    df["page_id"] = df["page"].map(lambda x: page_idx.get(x, 0)).astype(np.int64)  # Integer ID for page category
    df["metro_id"] = df["metro"].map(lambda x: metro_mapping.get(x, 0)).astype(np.int64)  # City/metro ID
    df["state_id"] = df["state"].map(lambda x: state_mapping.get(x, 0)).astype(np.int64)  # State/region ID
    df["device_id"] = df["device"].map(lambda x: device_mapping.get(x, 0)).astype(np.int64)  # Device type ID
    return df


def _infer_numeric_cols(df: pd.DataFrame) -> List[str]:
    exclude = {
        "userId",
        "song",
        "artist",
        "page",
        "metro",
        "state",
        "device",
        "regis_time",
        "time",
        "Cancellation Confirmation",
        "page_id",
        "prev_page_id",
        "metro_id",
        "state_id",
        "device_id",
    }
    numeric_cols = [
        c for c in df.columns if c not in exclude and not pd.api.types.is_object_dtype(df[c])
    ]
    return numeric_cols


def _build_cache_key(
    train_path: str,
    test_path: str,
    val_ratio: float,
    random_state: int,
    truncate_buffer_min: int,
    truncate_buffer_frac: float,
    cutoff_time: Optional[Union[pd.Timestamp, str, int]],
    drop_inactive_before_cutoff: bool,
) -> str:
    """
    Build a stable hash key for a given feature configuration so that we can
    cache and reuse computed datasets across runs.
    """
    key_payload = {
        "version": FEATURE_PIPELINE_VERSION,
        "train_path": str(Path(train_path).resolve()),
        "test_path": str(Path(test_path).resolve()),
        "val_ratio": float(val_ratio),
        "random_state": int(random_state),
        "truncate_buffer_min": int(truncate_buffer_min),
        "truncate_buffer_frac": float(truncate_buffer_frac),
        "cutoff_time": None if cutoff_time is None else str(cutoff_time),
        "drop_inactive_before_cutoff": bool(drop_inactive_before_cutoff),
    }
    key_str = json.dumps(key_payload, sort_keys=True)
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def prepare_datasets(
    train_path: str,
    test_path: str,
    val_ratio: float = 0.2,
    random_state: int = 42,
    truncate_buffer_min: int = 2,
    truncate_buffer_frac: float = 0.1,
    cutoff_time: Optional[Union[pd.Timestamp, str, int]] = None,
    drop_inactive_before_cutoff: bool = False,
    use_cache: bool = True,
    cache_dir: str = "feature_cache",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, FeatureArtifacts]:
    """
    Full pipeline: load data, feature engineer, encode categoricals, scale numerics,
    and split users into train/validation.
    - cutoff_time: str/timestamp keeps events before that absolute time; int drops each user's
      most recent `cutoff_time` days to train on earlier history and predict the held-out window.
    - use_cache: when True, cache the resulting datasets to disk keyed by the arguments so
      subsequent runs with the same configuration can be loaded instantly.
    """
    cache_path: Optional[Path] = None
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = _build_cache_key(
            train_path=train_path,
            test_path=test_path,
            val_ratio=val_ratio,
            random_state=random_state,
            truncate_buffer_min=truncate_buffer_min,
            truncate_buffer_frac=truncate_buffer_frac,
            cutoff_time=cutoff_time,
            drop_inactive_before_cutoff=drop_inactive_before_cutoff,
        )
        cache_path = Path(cache_dir) / f"features_{cache_key}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            return (
                cached["train_df"],
                cached["val_df"],
                cached["test_df"],
                cached["labels"],
                cached["artifacts"],
            )

    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)

    # Labels: whether the full original sequence ever contains Cancellation Confirmation (untruncated)
    labels = (
        train_raw.groupby("userId")["page"]
        .apply(lambda s: int((s == "Cancellation Confirmation").any()))
        .astype(int)
    )

    # Optional: only keep behaviors before a cutoff point when building features.
    if cutoff_time is not None:
        if isinstance(cutoff_time, Integral):
            train_raw = truncate_last_days_per_user(train_raw, int(cutoff_time))
            # test_raw = truncate_last_days_per_user(test_raw, int(cutoff_time))
            if drop_inactive_before_cutoff:
                active_users = train_raw["userId"].unique()
                labels = labels[labels.index.isin(active_users)]
        else:
            cutoff_ts = pd.to_datetime(cutoff_time)
            train_raw = train_raw[train_raw["time"] <= cutoff_ts].copy()
            if drop_inactive_before_cutoff:
                # Drop users who have no activity before the cutoff
                active_users = train_raw["userId"].unique()
                labels = labels[labels.index.isin(active_users)]

    # Remove Cancellation Confirmation rows from feature building to avoid leakage
    train_raw = train_raw[train_raw["page"] != "Cancellation Confirmation"].copy()
    train_raw = train_raw[train_raw["page"] != "Cancel"].copy()
    # test_raw = test_raw[test_raw["page"] != "Cancellation Confirmation"].copy()

    # Trim timelines to avoid using behaviors immediately before churn points
    train_raw = truncate_user_histories(
        train_raw, buffer_min=truncate_buffer_min, buffer_frac=truncate_buffer_frac
    )

    # Page categories exclude Cancellation Confirmation itself
    page_categories = sorted(set(train_raw["page"]).union(set(test_raw["page"])))

    train_fe = feature_engineer(train_raw, page_categories)
    test_fe = feature_engineer(test_raw, page_categories)

    metro_map = _build_mapping([train_fe["metro"], test_fe["metro"]])
    state_map = _build_mapping([train_fe["state"], test_fe["state"]])
    device_map = _build_mapping([train_fe["device"], test_fe["device"]])

    train_fe = encode_categoricals(train_fe, page_categories, metro_map, state_map, device_map)
    test_fe = encode_categoricals(test_fe, page_categories, metro_map, state_map, device_map)

    # Previous page ID within each user's sequence (0 = no previous page).
    train_fe["prev_page_id"] = (
        train_fe.groupby("userId")["page_id"].shift(1).fillna(0).astype(np.int64)
    )
    test_fe["prev_page_id"] = (
        test_fe.groupby("userId")["page_id"].shift(1).fillna(0).astype(np.int64)
    )

    train_fe["regis_time_seconds"] = train_fe["regis_time"].dt.total_seconds()  # Seconds since registration
    test_fe["regis_time_seconds"] = test_fe["regis_time"].dt.total_seconds()

    # Remove columns that may leak labels or explode dimensionality (raw categories, text, one-hots)
    drop_cols = set(page_categories) | {
        "page",
        "metro",
        "state",
        "device",
        "regis_time",
        "song",
        "artist",
    }
    train_fe = train_fe.drop(columns=[c for c in drop_cols if c in train_fe.columns])
    test_fe = test_fe.drop(columns=[c for c in drop_cols if c in test_fe.columns])

    numeric_cols = _infer_numeric_cols(train_fe) + ["regis_time_seconds"]
    numeric_cols = sorted(set(numeric_cols))

    # Ensure consistent column ordering and fill missing values
    for col in numeric_cols:
        if col not in train_fe:
            train_fe[col] = 0.0
        if col not in test_fe:
            test_fe[col] = 0.0

    train_fe[numeric_cols] = train_fe[numeric_cols].fillna(0)
    test_fe[numeric_cols] = test_fe[numeric_cols].fillna(0)

    unique_users = labels.index
    stratify = labels if labels.value_counts().min() >= 2 else None
    train_users, val_users = train_test_split(
        unique_users, test_size=val_ratio, random_state=random_state, stratify=stratify
    )
    
    train_df = train_fe[train_fe["userId"].isin(train_users)].copy()
    val_df = train_fe[train_fe["userId"].isin(val_users)].copy()

    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    test_fe[numeric_cols] = scaler.transform(test_fe[numeric_cols])

    artifacts = FeatureArtifacts(
        page_categories=list(page_categories),
        metro_mapping=metro_map,
        state_mapping=state_map,
        device_mapping=device_map,
        numeric_cols=numeric_cols,
        scaler=scaler,
    )
    if cache_path is not None:
        to_cache = {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_fe,
            "labels": labels,
            "artifacts": artifacts,
        }
        with cache_path.open("wb") as f:
            pickle.dump(to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_df, val_df, test_fe, labels, artifacts

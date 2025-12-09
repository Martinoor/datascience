"""
Reusable feature engineering pipeline for the churn dataset.

The logic mirrors the exploration done in `feature_engineering.ipynb` but is
packaged so it can be reused by training / inference scripts and applied to
both train and test consistently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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
    reduce leakage from behaviors紧贴退订前的尾段。
    """
    kept = []
    for _, g in df.sort_values(["userId", "time"]).groupby("userId", sort=False):
        L = len(g)
        buffer = max(buffer_min, int(L * buffer_frac))
        if L <= 1:
            kept.append(g)
            continue
        cutoff = max(L - buffer, 1)
        kept.append(g.iloc[:cutoff])
    return pd.concat(kept, ignore_index=True)


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

    df["num_distinct_song_until_now"] = user_song_count
    df["num_distinct_artist_until_now"] = user_artist_count
    df["song_top1_frac_until_now"] = song_top1_frac
    df["song_top1_count_until_now"] = song_top1_count
    df["song_top3_frac_until_now"] = song_top3_frac
    df["song_top3_count_until_now"] = song_top3_count
    df["artist_top1_frac_until_now"] = artist_top1_frac
    df["artist_top1_count_until_now"] = artist_top1_count
    df["artist_top3_frac_until_now"] = artist_top3_frac
    df["artist_top3_count_until_now"] = artist_top3_count

    df = df.drop(columns=["_song_valid", "_artist_valid"])
    return df


def _add_page_dummies(df: pd.DataFrame, page_categories: Sequence[str]) -> pd.DataFrame:
    cat_type = pd.api.types.CategoricalDtype(categories=page_categories)
    df["page"] = df["page"].astype(cat_type)
    dummies = pd.get_dummies(df["page"])
    df = pd.concat([df, dummies], axis=1)
    return df


def feature_engineer(df: pd.DataFrame, page_categories: Sequence[str]) -> pd.DataFrame:
    """
    Apply the feature engineering steps to a raw dataframe.
    """
    df = df.copy()
    to_drop = ["firstName", "lastName", "ts", "auth", "itemInSession", "sessionId", "method"]
    df = df.drop(columns=to_drop)
    df = df.sort_values(["userId", "time"])

    df["error occur"] = compute_error_ratio(df)
    df = df.drop(columns=["status"])

    df["gender"] = df["gender"].map({"F": 0, "M": 1}).fillna(0).astype(np.int64)
    df["level"] = df["level"].map({"free": 0, "paid": 1}).fillna(0).astype(np.int64)

    df = _add_page_dummies(df, page_categories)

    df[["metro", "state"]] = df["location"].str.rsplit(", ", n=1, expand=True)
    df = df.drop(columns=["location"])

    df["device"] = df["userAgent"].apply(detect_device)
    df = df.drop(columns=["userAgent"])

    df = get_song_stats_fast(df)
    df["regis_time"] = df.time - df.registration
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
    df["page_id"] = df["page"].map(lambda x: page_idx.get(x, 0)).astype(np.int64)
    df["metro_id"] = df["metro"].map(lambda x: metro_mapping.get(x, 0)).astype(np.int64)
    df["state_id"] = df["state"].map(lambda x: state_mapping.get(x, 0)).astype(np.int64)
    df["device_id"] = df["device"].map(lambda x: device_mapping.get(x, 0)).astype(np.int64)
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
        "metro_id",
        "state_id",
        "device_id",
    }
    numeric_cols = [
        c for c in df.columns if c not in exclude and not pd.api.types.is_object_dtype(df[c])
    ]
    return numeric_cols


def prepare_datasets(
    train_path: str,
    test_path: str,
    val_ratio: float = 0.2,
    random_state: int = 42,
    truncate_buffer_min: int = 2,
    truncate_buffer_frac: float = 0.1,
    cutoff_time: Optional[pd.Timestamp] = None,
    drop_inactive_before_cutoff: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, FeatureArtifacts]:
    """
    Full pipeline: load data, feature engineer, encode categoricals, scale numerics,
    and split users into train/validation.
    """
    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)

    # 标签：基于原始序列是否出现过 Cancellation Confirmation（全量，不受截断影响）
    labels = (
        train_raw.groupby("userId")["page"]
        .apply(lambda s: int((s == "Cancellation Confirmation").any()))
        .astype(int)
    )

    # 可选：仅使用 cutoff_time 之前的行为做特征
    if cutoff_time is not None:
        cutoff_ts = pd.to_datetime(cutoff_time)
        train_raw = train_raw[train_raw["time"] <= cutoff_ts].copy()
        if drop_inactive_before_cutoff:
            # 丢弃在 cutoff 前就没有任何记录的用户
            active_users = train_raw["userId"].unique()
            labels = labels[labels.index.isin(active_users)]

    # 训练/验证特征中移除 Cancellation Confirmation 行，避免泄漏
    train_raw = train_raw[train_raw["page"] != "Cancellation Confirmation"].copy()

    # 时间序列截断，避免使用过于靠近退订节点的尾部行为
    train_raw = truncate_user_histories(
        train_raw, buffer_min=truncate_buffer_min, buffer_frac=truncate_buffer_frac
    )

    # page 类别不包含 Cancellation Confirmation
    page_categories = sorted(set(train_raw["page"]).union(set(test_raw["page"])))

    train_fe = feature_engineer(train_raw, page_categories)
    test_fe = feature_engineer(test_raw, page_categories)

    metro_map = _build_mapping([train_fe["metro"], test_fe["metro"]])
    state_map = _build_mapping([train_fe["state"], test_fe["state"]])
    device_map = _build_mapping([train_fe["device"], test_fe["device"]])

    train_fe = encode_categoricals(train_fe, page_categories, metro_map, state_map, device_map)
    test_fe = encode_categoricals(test_fe, page_categories, metro_map, state_map, device_map)

    train_fe["regis_time_seconds"] = train_fe["regis_time"].dt.total_seconds()
    test_fe["regis_time_seconds"] = test_fe["regis_time"].dt.total_seconds()

    # 严格去除可能导致泄露或维度膨胀的列（原始类别、文本、page one-hot 等）
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
    return train_df, val_df, test_fe, labels, artifacts

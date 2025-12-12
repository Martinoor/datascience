from __future__ import annotations

"""
Daily (user × day) feature pipeline for churn prediction.

This is intentionally isolated under `daily_feature/` so the existing event-level
pipeline (`feature_pipeline.py`) can continue to evolve independently.
"""

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_pipeline import detect_device, truncate_last_days_per_user

DAILY_FEATURE_PIPELINE_VERSION = "0.1"

LEAK_PAGES = {"Cancel", "Cancellation Confirmation"}
LEAK_AUTH_VALUES = {"Cancelled"}


DAILY_PAGE_COUNT_PAGES: List[str] = [
    # Engagement / interaction
    "Thumbs Up",
    "Thumbs Down",
    "Add Friend",
    "Add to Playlist",
    # Experience
    "Roll Advert",
    "Error",
    # Intent / support
    "Help",
    "Settings",
    "Save Settings",
    "About",
    "Home",
    "Logout",
    # Subscription intent
    "Upgrade",
    "Submit Upgrade",
    "Downgrade",
    "Submit Downgrade",
    # Test-only auth pages (keep as a single bucket to avoid train/test mismatch)
    "Login",
    "Register",
    "Submit Registration",
]


def _safe_name(raw: str) -> str:
    name = str(raw).strip().lower()
    for ch in [" ", "/", "-", "(", ")", "[", "]"]:
        name = name.replace(ch, "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def _build_mapping(series_list: Sequence[pd.Series]) -> Dict[str, int]:
    values: List[str] = []
    for s in series_list:
        values.extend(list(s.dropna().unique()))
    categories = sorted(set(values))
    return {cat: idx + 1 for idx, cat in enumerate(categories)}  # 0 = unknown/pad


@dataclass
class DailyFeatureArtifacts:
    metro_mapping: Dict[str, int]
    state_mapping: Dict[str, int]
    device_mapping: Dict[str, int]
    numeric_cols: List[str]
    scaler: StandardScaler
    page_count_pages: List[str]
    calendar_min_day: pd.Timestamp
    calendar_max_day: pd.Timestamp


def _build_cache_key(
    train_path: str,
    test_path: str,
    val_ratio: float,
    random_state: int,
    cutoff_time: Optional[Union[pd.Timestamp, str, int]],
    drop_inactive_before_cutoff: bool,
    calendar_strategy: str,
    include_song_artist: bool,
) -> str:
    payload = {
        "version": DAILY_FEATURE_PIPELINE_VERSION,
        "train_path": str(Path(train_path).resolve()),
        "test_path": str(Path(test_path).resolve()),
        "val_ratio": float(val_ratio),
        "random_state": int(random_state),
        "cutoff_time": None if cutoff_time is None else str(cutoff_time),
        "drop_inactive_before_cutoff": bool(drop_inactive_before_cutoff),
        "calendar_strategy": str(calendar_strategy),
        "include_song_artist": bool(include_song_artist),
        "page_count_pages": list(DAILY_PAGE_COUNT_PAGES),
    }
    key_str = json.dumps(payload, sort_keys=True)
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def _filter_leakage_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["page"].isin(LEAK_PAGES)].copy()
    if "auth" in df.columns:
        df = df[~df["auth"].isin(LEAK_AUTH_VALUES)].copy()
    return df


def _build_user_static(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-user static fields (stable across time)."""
    needed = ["userId", "gender", "registration", "location", "userAgent"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for user static: {missing}")

    static = (
        df.groupby("userId", sort=False)[["gender", "registration", "location", "userAgent"]]
        .first()
        .reset_index()
    )
    static["gender"] = static["gender"].map({"F": 0, "M": 1}).fillna(0).astype(np.int64)
    static["registration_day"] = static["registration"].dt.floor("D")

    static[["metro", "state"]] = static["location"].str.rsplit(", ", n=1, expand=True)
    static["device"] = static["userAgent"].apply(detect_device)
    static = static.drop(columns=["registration", "location", "userAgent"])
    return static


def _encode_static_categories(
    train_static: pd.DataFrame, test_static: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, int], Dict[str, int]]:
    metro_map = _build_mapping([train_static["metro"], test_static["metro"]])
    state_map = _build_mapping([train_static["state"], test_static["state"]])
    device_map = _build_mapping([train_static["device"], test_static["device"]])

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["metro_id"] = out["metro"].map(lambda x: metro_map.get(x, 0)).astype(np.int64)
        out["state_id"] = out["state"].map(lambda x: state_map.get(x, 0)).astype(np.int64)
        out["device_id"] = out["device"].map(lambda x: device_map.get(x, 0)).astype(np.int64)
        return out.drop(columns=["metro", "state", "device"])

    return _apply(train_static), _apply(test_static), metro_map, state_map, device_map


def _compute_daily_aggregates(
    df: pd.DataFrame,
    *,
    page_count_pages: Sequence[str],
    include_song_artist: bool,
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    df = df.copy()
    df["day"] = df["time"].dt.floor("D")
    df["hour"] = df["time"].dt.hour.astype(np.int16)

    cal_min = df["day"].min()
    cal_max = df["day"].max()

    is_nextsong = (df["page"] == "NextSong").astype(np.int8)
    is_put = (df["method"] == "PUT").astype(np.int8) if "method" in df.columns else None
    is_get = (df["method"] == "GET").astype(np.int8) if "method" in df.columns else None

    g = df.groupby(["userId", "day"], sort=False)
    daily = g.agg(
        d_events=("page", "size"),
        d_pages_unique=("page", "nunique"),
        d_sessions=("sessionId", "nunique"),
        d_active_hours=("hour", "nunique"),
        _t_min=("time", "min"),
        _t_max=("time", "max"),
    )
    daily["d_first_hour"] = daily["_t_min"].dt.hour.astype(np.int16)
    daily["d_last_hour"] = daily["_t_max"].dt.hour.astype(np.int16)
    daily["d_time_span_sec"] = (
        (daily["_t_max"] - daily["_t_min"]).dt.total_seconds().fillna(0).astype(np.float32)
    )
    daily = daily.drop(columns=["_t_min", "_t_max"])

    if is_put is not None:
        daily["d_put_cnt"] = is_put.groupby([df["userId"], df["day"]]).sum().astype(np.int32)
    else:
        daily["d_put_cnt"] = 0
    if is_get is not None:
        daily["d_get_cnt"] = is_get.groupby([df["userId"], df["day"]]).sum().astype(np.int32)
    else:
        daily["d_get_cnt"] = 0

    is_404 = (df["status"] == 404).astype(np.int8) if "status" in df.columns else None
    is_307 = (df["status"] == 307).astype(np.int8) if "status" in df.columns else None
    if is_404 is not None:
        daily["d_status_404_cnt"] = is_404.groupby([df["userId"], df["day"]]).sum().astype(np.int32)
    else:
        daily["d_status_404_cnt"] = 0
    if is_307 is not None:
        daily["d_status_307_cnt"] = is_307.groupby([df["userId"], df["day"]]).sum().astype(np.int32)
    else:
        daily["d_status_307_cnt"] = 0
    daily["d_error_page_cnt"] = (
        (df["page"] == "Error").astype(np.int8).groupby([df["userId"], df["day"]]).sum().astype(np.int32)
    )

    # Page counts for a stable list of pages.
    page_count_pages = list(page_count_pages)
    page_df = df[df["page"].isin(page_count_pages)][["userId", "day", "page"]]
    if len(page_df) > 0:
        page_counts = page_df.groupby(["userId", "day", "page"], sort=False).size().unstack(fill_value=0)
        page_counts = page_counts.rename(columns={p: f"d_page_{_safe_name(p)}_cnt" for p in page_counts.columns})
        daily = daily.join(page_counts, how="left")

    # NextSong-related.
    ns = df[is_nextsong == 1]
    if len(ns) > 0:
        ns_g = ns.groupby(["userId", "day"], sort=False)
        daily["d_nextsong_cnt"] = ns_g.size().astype(np.int32)
        daily["d_listen_sec_sum"] = ns_g["length"].sum(min_count=1).fillna(0).astype(np.float32)
        daily["d_listen_sec_mean"] = ns_g["length"].mean().fillna(0).astype(np.float32)
        daily["d_listen_sec_std"] = ns_g["length"].std(ddof=0).fillna(0).astype(np.float32)

        if include_song_artist:
            daily["d_song_unique"] = ns_g["song"].nunique(dropna=True).astype(np.int32)
            daily["d_artist_unique"] = ns_g["artist"].nunique(dropna=True).astype(np.int32)

            # top1 artist concentration
            artist_counts = (
                ns.dropna(subset=["artist"])
                .groupby(["userId", "day", "artist"], sort=False)
                .size()
            )
            if len(artist_counts) > 0:
                top1 = artist_counts.groupby(["userId", "day"], sort=False).max()
                daily["d_artist_top1_cnt"] = top1.astype(np.int32)
            else:
                daily["d_artist_top1_cnt"] = 0
        else:
            daily["d_song_unique"] = 0
            daily["d_artist_unique"] = 0
            daily["d_artist_top1_cnt"] = 0
    else:
        daily["d_nextsong_cnt"] = 0
        daily["d_listen_sec_sum"] = 0.0
        daily["d_listen_sec_mean"] = 0.0
        daily["d_listen_sec_std"] = 0.0
        daily["d_song_unique"] = 0
        daily["d_artist_unique"] = 0
        daily["d_artist_top1_cnt"] = 0

    # Level daily features.
    paid_flag = df["level"].map({"free": 0, "paid": 1}).fillna(0).astype(np.int8)
    daily["d_paid_ratio"] = paid_flag.groupby([df["userId"], df["day"]]).mean().astype(np.float32)
    level_nunique = paid_flag.groupby([df["userId"], df["day"]]).nunique().astype(np.int16)
    daily["d_level_changed_in_day"] = (level_nunique > 1).astype(np.int8)

    idx_last = df.groupby(["userId", "day"], sort=False)["time"].idxmax()
    last_paid = (
        df.loc[idx_last, ["userId", "day"]]
        .assign(_paid=paid_flag.loc[idx_last].astype(np.int8).to_numpy())
        .set_index(["userId", "day"])["_paid"]
    )
    daily = daily.join(last_paid.rename("d_level_last"), how="left")
    daily["d_level_last"] = daily["d_level_last"].fillna(0).astype(np.int8)

    # Session-level then aggregate back to day.
    sess_g = df.groupby(["userId", "day", "sessionId"], sort=False)
    sess = sess_g.agg(
        s_events=("page", "size"),
        _t_min=("time", "min"),
        _t_max=("time", "max"),
    )
    sess["s_nextsong"] = (
        is_nextsong.groupby([df["userId"], df["day"], df["sessionId"]]).sum().astype(np.int32)
    )
    sess["s_duration_sec"] = (sess["_t_max"] - sess["_t_min"]).dt.total_seconds().fillna(0).astype(np.float32)
    sess = sess.drop(columns=["_t_min", "_t_max"])

    sess_day = sess.reset_index().groupby(["userId", "day"], sort=False)
    daily_sess = sess_day.agg(
        d_sess_events_mean=("s_events", "mean"),
        d_sess_events_max=("s_events", "max"),
        d_sess_duration_mean=("s_duration_sec", "mean"),
        d_sess_duration_max=("s_duration_sec", "max"),
        d_sess_nextsong_mean=("s_nextsong", "mean"),
        d_sess_nextsong_max=("s_nextsong", "max"),
    )
    daily = daily.join(daily_sess, how="left")

    # P90 (optional but useful); group sizes are small so this is OK.
    daily["d_sess_events_p90"] = sess_day["s_events"].quantile(0.9).astype(np.float32)
    daily["d_sess_duration_p90"] = sess_day["s_duration_sec"].quantile(0.9).astype(np.float32)

    daily["d_events_per_session"] = (
        daily["d_events"].astype(np.float32) / daily["d_sessions"].replace(0, np.nan).astype(np.float32)
    ).fillna(0.0)

    # Ratios that require divisors.
    with np.errstate(divide="ignore", invalid="ignore"):
        daily["d_repeat_song_rate"] = (
            1.0
            - daily["d_song_unique"].astype(np.float32)
            / daily["d_nextsong_cnt"].replace(0, np.nan).astype(np.float32)
        ).fillna(0.0)
        daily["d_artist_concentration"] = (
            daily["d_artist_top1_cnt"].astype(np.float32)
            / daily["d_nextsong_cnt"].replace(0, np.nan).astype(np.float32)
        ).fillna(0.0)
        daily["d_error_rate"] = (
            (daily["d_status_404_cnt"] + daily["d_error_page_cnt"]).astype(np.float32)
            / daily["d_events"].replace(0, np.nan).astype(np.float32)
        ).fillna(0.0)
        daily["d_get_rate"] = (
            daily["d_get_cnt"].astype(np.float32) / daily["d_events"].replace(0, np.nan).astype(np.float32)
        ).fillna(0.0)

    daily["d_long_session_flag"] = (daily["d_sess_duration_max"].fillna(0) > 7200).astype(np.int8)
    daily = daily.fillna(0)
    return daily.reset_index(), cal_min, cal_max


def _add_cross_day_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values(["userId", "day"]).reset_index(drop=True)

    delta_days = daily.groupby("userId")["day"].diff().dt.days
    daily["days_since_last_active"] = delta_days.fillna(0).astype(np.int16)

    is_break = daily["days_since_last_active"] != 1
    streak_id = is_break.groupby(daily["userId"]).cumsum()
    daily["active_streak_days"] = (
        daily.groupby(["userId", streak_id]).cumcount().astype(np.int16) + 1
    )

    daily["day_of_week"] = daily["day"].dt.dayofweek.astype(np.int16)
    daily["dow_id"] = (daily["day_of_week"] + 1).astype(np.int64)  # 0 reserved for pad

    # Rolling stats on a few high-signal counters.
    for col, out_col in [
        ("d_events", "roll7_d_events_mean"),
        ("d_nextsong_cnt", "roll7_d_nextsong_mean"),
        ("d_sessions", "roll7_d_sessions_mean"),
    ]:
        roll = (
            daily.groupby("userId", sort=False)[col]
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        daily[out_col] = roll.astype(np.float32)
        prev = (
            daily.groupby("userId", sort=False)[out_col]
            .shift(7)
            .fillna(daily[out_col])
            .astype(np.float32)
        )
        daily[f"trend7_{col}"] = (daily[out_col] - prev).astype(np.float32)

    return daily


def _apply_calendar_strategy(
    daily: pd.DataFrame,
    all_user_ids: Sequence[str],
    cal_min: pd.Timestamp,
    cal_max: pd.Timestamp,
    *,
    calendar_strategy: str,
) -> pd.DataFrame:
    strategy = str(calendar_strategy).lower()
    if strategy == "active":
        return daily
    if strategy != "full":
        raise ValueError(f"Unknown calendar_strategy={calendar_strategy!r}; use 'active' or 'full'.")

    calendar = pd.date_range(cal_min, cal_max, freq="D")
    base_index = pd.MultiIndex.from_product(
        [list(all_user_ids), calendar], names=["userId", "day"]
    )
    daily_idxed = daily.set_index(["userId", "day"])
    expanded = daily_idxed.reindex(base_index).reset_index()
    # Missing days are inactive by definition; fill with 0 (tenure/dow added later).
    expanded = expanded.fillna(0)
    return expanded


def prepare_daily_datasets(
    train_path: str,
    test_path: str,
    *,
    val_ratio: float = 0.2,
    random_state: int = 42,
    cutoff_time: Optional[Union[pd.Timestamp, str, int]] = None,
    drop_inactive_before_cutoff: bool = False,
    calendar_strategy: str = "active",  # "active" | "full"
    include_song_artist: bool = True,
    use_cache: bool = True,
    cache_dir: str = "daily_feature/cache",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, DailyFeatureArtifacts]:
    cache_path: Optional[Path] = None
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = _build_cache_key(
            train_path=train_path,
            test_path=test_path,
            val_ratio=val_ratio,
            random_state=random_state,
            cutoff_time=cutoff_time,
            drop_inactive_before_cutoff=drop_inactive_before_cutoff,
            calendar_strategy=calendar_strategy,
            include_song_artist=include_song_artist,
        )
        cache_path = Path(cache_dir) / f"daily_features_{cache_key}.pkl"
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

    cols = [
        "userId",
        "time",
        "page",
        "sessionId",
        "status",
        "method",
        "level",
        "gender",
        "registration",
        "location",
        "userAgent",
        "song",
        "artist",
        "length",
        "auth",
    ]
    train_raw = pd.read_parquet(train_path, columns=cols)
    test_raw = pd.read_parquet(test_path, columns=cols)

    labels = (
        train_raw.groupby("userId")["page"]
        .apply(lambda s: int((s == "Cancellation Confirmation").any()))
        .astype(int)
    )

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

    train_static = _build_user_static(train_raw)
    test_static = _build_user_static(test_raw)
    train_static, test_static, metro_map, state_map, device_map = _encode_static_categories(
        train_static, test_static
    )

    train_feat_raw = _filter_leakage_events(train_raw)
    test_feat_raw = _filter_leakage_events(test_raw)

    train_daily, cal_min_train, cal_max_train = _compute_daily_aggregates(
        train_feat_raw, page_count_pages=DAILY_PAGE_COUNT_PAGES, include_song_artist=include_song_artist
    )
    test_daily, cal_min_test, cal_max_test = _compute_daily_aggregates(
        test_feat_raw, page_count_pages=DAILY_PAGE_COUNT_PAGES, include_song_artist=include_song_artist
    )

    cal_min = min(cal_min_train, cal_min_test)
    cal_max = max(cal_max_train, cal_max_test)

    # Calendar expansion (optional).
    train_daily = _apply_calendar_strategy(
        train_daily,
        all_user_ids=train_daily["userId"].unique(),
        cal_min=cal_min,
        cal_max=cal_max,
        calendar_strategy=calendar_strategy,
    )
    test_daily = _apply_calendar_strategy(
        test_daily,
        all_user_ids=test_daily["userId"].unique(),
        cal_min=cal_min,
        cal_max=cal_max,
        calendar_strategy=calendar_strategy,
    )

    # Merge static features (gender/tenure + categorical IDs).
    train_daily = train_daily.merge(train_static, on="userId", how="left")
    test_daily = test_daily.merge(test_static, on="userId", how="left")

    for df in (train_daily, test_daily):
        df["registration_day"] = pd.to_datetime(df["registration_day"])
        df["tenure_days"] = (df["day"] - df["registration_day"]).dt.days.fillna(0).astype(np.int32)

    train_daily = _add_cross_day_features(train_daily)
    test_daily = _add_cross_day_features(test_daily)

    # log1p on heavy-tailed count/duration features.
    log1p_cols = [
        c
        for c in train_daily.columns
        if c.startswith("d_")
        and (
            c.endswith("_cnt")
            or c.endswith("_sum")
            or c.endswith("_sec")
            or c in {"d_events", "d_sessions", "d_pages_unique", "d_active_hours", "d_nextsong_cnt"}
            or c.startswith("d_sess_")
        )
    ] + [
        "days_since_last_active",
        "active_streak_days",
        "tenure_days",
        "roll7_d_events_mean",
        "roll7_d_nextsong_mean",
        "roll7_d_sessions_mean",
    ]
    log1p_cols = sorted(set([c for c in log1p_cols if c in train_daily.columns]))
    for df in (train_daily, test_daily):
        for c in log1p_cols:
            df[c] = np.log1p(df[c].astype(np.float32)).astype(np.float32)

    train_daily = train_daily.fillna(0)
    test_daily = test_daily.fillna(0)

    exclude = {
        "userId",
        "day",
        "registration_day",
        "metro_id",
        "state_id",
        "device_id",
        "dow_id",
        "day_of_week",
    }
    numeric_cols = [
        c
        for c in train_daily.columns
        if c not in exclude and not pd.api.types.is_object_dtype(train_daily[c])
    ]
    numeric_cols = sorted(set(numeric_cols))

    # Align any missing columns across splits.
    for col in numeric_cols:
        if col not in train_daily:
            train_daily[col] = 0.0
        if col not in test_daily:
            test_daily[col] = 0.0

    # Only keep users that have daily rows.
    active_users = pd.Index(train_daily["userId"].unique())
    labels = labels[labels.index.isin(active_users)]

    unique_users = labels.index
    stratify = labels if labels.value_counts().min() >= 2 else None
    train_users, val_users = train_test_split(
        unique_users, test_size=val_ratio, random_state=random_state, stratify=stratify
    )

    train_df = train_daily[train_daily["userId"].isin(train_users)].copy()
    val_df = train_daily[train_daily["userId"].isin(val_users)].copy()

    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols]).astype(np.float32)
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols]).astype(np.float32)
    test_daily[numeric_cols] = scaler.transform(test_daily[numeric_cols]).astype(np.float32)

    artifacts = DailyFeatureArtifacts(
        metro_mapping=metro_map,
        state_mapping=state_map,
        device_mapping=device_map,
        numeric_cols=numeric_cols,
        scaler=scaler,
        page_count_pages=list(DAILY_PAGE_COUNT_PAGES),
        calendar_min_day=cal_min,
        calendar_max_day=cal_max,
    )

    if cache_path is not None:
        to_cache = {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_daily,
            "labels": labels,
            "artifacts": artifacts,
        }
        with cache_path.open("wb") as f:
            pickle.dump(to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_df, val_df, test_daily, labels, artifacts

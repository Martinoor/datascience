from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CutoffTime = Optional[Union[str, int]]


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


def _slugify(value: str) -> str:
    value = value.strip().lower()
    out = []
    prev_underscore = False
    for ch in value:
        is_alnum = ("a" <= ch <= "z") or ("0" <= ch <= "9")
        if is_alnum:
            out.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                out.append("_")
                prev_underscore = True
    slug = "".join(out).strip("_")
    return slug or "unknown"


def _file_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _hash_to_unit_interval(text: str) -> float:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return value / 2**64


def _parse_cutoff_time(value: str | None) -> CutoffTime:
    if value is None:
        return None
    v = value.strip()
    if v == "" or v.lower() in {"none", "null"}:
        return None
    if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
        return int(v)
    return v


def _to_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


@dataclass(frozen=True)
class FeatureBuildConfig:
    cutoff_time: CutoffTime = None
    horizon_days: int = 10
    label_mode: Literal["horizon", "ever"] = "ever"
    drop_last_k_events: int = 0
    exclude_pages: tuple[str, ...] = ("Cancellation Confirmation", "Cancel")
    exclude_auth_values: tuple[str, ...] = ("Cancelled",)
    full_calendar: bool = True
    batch_size: int = 1_000_000

    def normalized(self) -> "FeatureBuildConfig":
        cutoff_time = self.cutoff_time
        horizon_days = int(self.horizon_days)
        label_mode: Literal["horizon", "ever"] = self.label_mode
        if isinstance(cutoff_time, int) and label_mode == "horizon":
            horizon_days = int(cutoff_time)
        return FeatureBuildConfig(
            cutoff_time=cutoff_time,
            horizon_days=horizon_days,
            label_mode=label_mode,
            drop_last_k_events=int(self.drop_last_k_events),
            exclude_pages=tuple(self.exclude_pages),
            exclude_auth_values=tuple(self.exclude_auth_values),
            full_calendar=bool(self.full_calendar),
            batch_size=int(self.batch_size),
        )


@dataclass(frozen=True)
class SplitConfig:
    val_prob_low: float = 0.8
    val_prob_high: float = 1.0
    split_seed: str = "split_v1"


def _format_float_for_path(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def make_run_name(
    feature_cfg: FeatureBuildConfig,
    split_cfg: SplitConfig,
    prefix: str = "user_day",
) -> str:
    feature_cfg = feature_cfg.normalized()
    feature_payload = asdict(feature_cfg)
    feature_payload.pop("batch_size", None)
    cutoff = feature_cfg.cutoff_time
    if cutoff is None:
        cutoff_tag = "cutoff_none"
    elif isinstance(cutoff, int):
        cutoff_tag = f"cutoff_tail{cutoff}d"
    else:
        cutoff_ts = _to_timestamp(cutoff)
        cutoff_tag = f"cutoff_{cutoff_ts.strftime('%Y%m%d')}"

    cal_tag = "calfull" if feature_cfg.full_calendar else "calsparse"
    val_tag = f"val{_format_float_for_path(split_cfg.val_prob_low)}-{_format_float_for_path(split_cfg.val_prob_high)}"
    seed_tag = _slugify(split_cfg.split_seed)
    key = _stable_hash(
        {
            "feature_cfg": feature_payload,
            "split_cfg": asdict(split_cfg),
            # Bump when feature schema changes.
            "version": 3,
        }
    )[:12]
    return (
        f"{prefix}_{cutoff_tag}_{feature_cfg.label_mode}_k{feature_cfg.drop_last_k_events}"
        f"_{cal_tag}_{val_tag}_{seed_tag}_{key}"
    )


def suggest_output_dir(
    feature_cfg: FeatureBuildConfig,
    split_cfg: SplitConfig,
    base_dir: Path = Path("data/processed"),
    prefix: str = "user_day",
) -> Path:
    return base_dir / make_run_name(feature_cfg=feature_cfg, split_cfg=split_cfg, prefix=prefix)


@dataclass(frozen=True)
class DatasetPaths:
    train_path: Path
    test_path: Path


@dataclass(frozen=True)
class BuildArtifacts:
    vocab: dict[str, list[object]]
    calendar_start_day: str
    calendar_num_days: int
    train_features_path: Path
    train_labels_path: Path
    test_features_path: Path


def scan_train_metadata(
    train_path: Path,
    batch_size: int,
    *,
    verbose: bool = False,
    log_every_batches: int = 5,
) -> tuple[
    list[str],
            "version": 4,
    dict[str, pd.Timestamp],
    pd.Timestamp,
    dict[str, pd.Timestamp],
    set[str],
    set[str],
    set[str],
    set[int],
]:
    pf = pq.ParquetFile(train_path)
    cols = ["userId", "time", "page", "auth", "method", "status", "registration"]
    user_ids: set[str] = set()
    end_time_by_user: dict[str, pd.Timestamp] = {}
    churn_time_by_user: dict[str, pd.Timestamp] = {}
    registration_by_user: dict[str, pd.Timestamp] = {}
    pages: set[str] = set()
    auth_values: set[str] = set()
    methods: set[str] = set()
    statuses: set[int] = set()
    global_max_time: pd.Timestamp | None = None

    rows_seen = 0
    batches_seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols, use_threads=True):
        batches_seen += 1
        rows_seen += batch.num_rows
        user_arr = batch.column("userId").to_numpy(zero_copy_only=False)
        time_arr = batch.column("time").to_numpy(zero_copy_only=False)
        page_arr = batch.column("page").to_numpy(zero_copy_only=False)
        auth_arr = batch.column("auth").to_numpy(zero_copy_only=False)
        method_arr = batch.column("method").to_numpy(zero_copy_only=False)
        status_arr = batch.column("status").to_numpy(zero_copy_only=False)
        reg_arr = batch.column("registration").to_numpy(zero_copy_only=False)

        user_ids.update(np.unique(user_arr).tolist())
        pages.update(np.unique(page_arr).tolist())
        auth_values.update(np.unique(auth_arr).tolist())
        methods.update(np.unique(method_arr).tolist())
        statuses.update(np.unique(status_arr).astype(int).tolist())

        # Vectorized per-user max time within batch, then merge dicts.
        df_tmp = pd.DataFrame({"userId": user_arr, "time": time_arr})
        batch_max = df_tmp.groupby("userId", sort=False)["time"].max()
        for uid, t in batch_max.items():
            prev = end_time_by_user.get(uid)
            if prev is None or t > prev:
                end_time_by_user[uid] = pd.Timestamp(t)

        # Churn time (Cancellation Confirmation) per user.
        mask_churn = page_arr == "Cancellation Confirmation"
        if mask_churn.any():
            churn_users = user_arr[mask_churn]
            churn_times = time_arr[mask_churn]
            df_churn = pd.DataFrame({"userId": churn_users, "time": churn_times})
            batch_min = df_churn.groupby("userId", sort=False)["time"].min()
            for uid, t in batch_min.items():
                ts = pd.Timestamp(t)
                prev = churn_time_by_user.get(uid)
                if prev is None or ts < prev:
                    churn_time_by_user[uid] = ts

        bmax = pd.Timestamp(time_arr.max())
        global_max_time = bmax if global_max_time is None else max(global_max_time, bmax)

        # Registration timestamp per user (static). We only need one value per user.
        if len(registration_by_user) < 25_000:
            reg_mask = pd.notna(reg_arr)
        else:
            reg_mask = None
        if reg_mask is not None and reg_mask.any():
            df_reg = pd.DataFrame({"userId": user_arr[reg_mask], "registration": reg_arr[reg_mask]})
            batch_reg = df_reg.groupby("userId", sort=False)["registration"].first()
            for uid, t in batch_reg.items():
                if uid not in registration_by_user:
                    registration_by_user[uid] = pd.Timestamp(t)

        if verbose and log_every_batches > 0 and (batches_seen % log_every_batches == 0):
            _log(verbose, f"[scan_train] batches={batches_seen} rows={rows_seen:,}")

    return (
        sorted(user_ids),
        end_time_by_user,
        churn_time_by_user,
        global_max_time or pd.Timestamp("1970-01-01"),
        registration_by_user,
        pages,
        auth_values,
        methods,
        statuses,
    )


def scan_test_metadata(
    test_path: Path,
    batch_size: int,
    *,
    verbose: bool = False,
    log_every_batches: int = 5,
) -> tuple[
    list[str],
    dict[str, pd.Timestamp],
    pd.Timestamp,
    dict[str, pd.Timestamp],
    set[str],
    set[str],
    set[str],
    set[int],
]:
    pf = pq.ParquetFile(test_path)
    cols = ["userId", "time", "page", "auth", "method", "status", "registration"]
    user_ids: set[str] = set()
    end_time_by_user: dict[str, pd.Timestamp] = {}
    registration_by_user: dict[str, pd.Timestamp] = {}
    pages: set[str] = set()
    auth_values: set[str] = set()
    methods: set[str] = set()
    statuses: set[int] = set()
    global_max_time: pd.Timestamp | None = None

    rows_seen = 0
    batches_seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols, use_threads=True):
        batches_seen += 1
        rows_seen += batch.num_rows
        user_arr = batch.column("userId").to_numpy(zero_copy_only=False)
        time_arr = batch.column("time").to_numpy(zero_copy_only=False)
        page_arr = batch.column("page").to_numpy(zero_copy_only=False)
        auth_arr = batch.column("auth").to_numpy(zero_copy_only=False)
        method_arr = batch.column("method").to_numpy(zero_copy_only=False)
        status_arr = batch.column("status").to_numpy(zero_copy_only=False)
        reg_arr = batch.column("registration").to_numpy(zero_copy_only=False)

        user_ids.update(np.unique(user_arr).tolist())
        pages.update(np.unique(page_arr).tolist())
        auth_values.update(np.unique(auth_arr).tolist())
        methods.update(np.unique(method_arr).tolist())
        statuses.update(np.unique(status_arr).astype(int).tolist())

        df_tmp = pd.DataFrame({"userId": user_arr, "time": time_arr})
        batch_max = df_tmp.groupby("userId", sort=False)["time"].max()
        for uid, t in batch_max.items():
            prev = end_time_by_user.get(uid)
            if prev is None or t > prev:
                end_time_by_user[uid] = pd.Timestamp(t)

        bmax = pd.Timestamp(time_arr.max())
        global_max_time = bmax if global_max_time is None else max(global_max_time, bmax)

        if len(registration_by_user) < 5_000:
            reg_mask = pd.notna(reg_arr)
        else:
            reg_mask = None
        if reg_mask is not None and reg_mask.any():
            df_reg = pd.DataFrame({"userId": user_arr[reg_mask], "registration": reg_arr[reg_mask]})
            batch_reg = df_reg.groupby("userId", sort=False)["registration"].first()
            for uid, t in batch_reg.items():
                if uid not in registration_by_user:
                    registration_by_user[uid] = pd.Timestamp(t)

        if verbose and log_every_batches > 0 and (batches_seen % log_every_batches == 0):
            _log(verbose, f"[scan_test] batches={batches_seen} rows={rows_seen:,}")

    return (
        sorted(user_ids),
        end_time_by_user,
        global_max_time or pd.Timestamp("1970-01-01"),
        registration_by_user,
        pages,
        auth_values,
        methods,
        statuses,
    )


def _build_calendar(start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.datetime64, int]:
    start_day = np.datetime64(start.normalize().to_datetime64(), "D")
    end_day = np.datetime64(end.normalize().to_datetime64(), "D")
    num_days = int((end_day - start_day).astype(int)) + 1
    return start_day, num_days


def _compute_cutoff_by_user_us(
    user_ids: list[str],
    end_time_by_user: dict[str, pd.Timestamp],
    cutoff_time: CutoffTime,
) -> tuple[np.ndarray | None, np.int64 | None]:
    if cutoff_time is None:
        return None, None
    if isinstance(cutoff_time, str):
        cutoff_us = np.int64(np.datetime64(_to_timestamp(cutoff_time).to_datetime64(), "us").astype("int64"))
        return None, cutoff_us
    if isinstance(cutoff_time, int):
        tail_days = int(cutoff_time)
        if tail_days <= 0:
            return None, None
        cutoff_us_by_user = np.empty(len(user_ids), dtype=np.int64)
        delta_us = np.int64(pd.Timedelta(days=tail_days).to_timedelta64().astype("timedelta64[us]").astype("int64"))
        for i, uid in enumerate(user_ids):
            end_ts = end_time_by_user[uid]
            end_us = np.int64(np.datetime64(end_ts.to_datetime64(), "us").astype("int64"))
            cutoff_us_by_user[i] = end_us - delta_us
        return cutoff_us_by_user, None
    raise TypeError(f"Unsupported cutoff_time type: {type(cutoff_time)}")


def build_train_labels(
    train_user_ids: list[str],
    churn_time_by_user: dict[str, pd.Timestamp],
    cutoff_by_user_us: np.ndarray | None,
    global_cutoff_us: np.int64 | None,
    horizon_days: int,
    label_mode: Literal["horizon", "ever"],
) -> pd.DataFrame:
    if label_mode == "horizon" and (cutoff_by_user_us is None and global_cutoff_us is None):
        raise ValueError("label_mode='horizon' requires a non-empty cutoff_time")
    horizon_us = np.int64(pd.Timedelta(days=int(horizon_days)).to_timedelta64().astype("timedelta64[us]").astype("int64"))
    labels = np.zeros(len(train_user_ids), dtype=np.int8)
    cutoff_us_out = np.full(len(train_user_ids), -1, dtype=np.int64)

    if cutoff_by_user_us is not None:
        cutoff_us_out = cutoff_by_user_us.copy()
    elif global_cutoff_us is not None:
        cutoff_us_out.fill(int(global_cutoff_us))

    for i, uid in enumerate(train_user_ids):
        churn_ts = churn_time_by_user.get(uid)
        if churn_ts is None:
            continue
        churn_us = np.int64(np.datetime64(churn_ts.to_datetime64(), "us").astype("int64"))
        if label_mode == "ever":
            labels[i] = 1
            continue
        c_us = cutoff_us_out[i]
        if c_us < 0:
            continue
        if churn_us >= c_us and churn_us <= c_us + horizon_us:
            labels[i] = 1

    df = pd.DataFrame(
        {
            "userId": train_user_ids,
            "label": labels,
        }
    )
    if (cutoff_by_user_us is not None) or (global_cutoff_us is not None):
        df["cutoff_time"] = pd.to_datetime(cutoff_us_out, unit="us")
        df["horizon_days"] = int(horizon_days)
    return df


def compute_drop_last_k_event_ids(
    parquet_path: Path,
    user_ids: list[str],
    cutoff_by_user_us: np.ndarray | None,
    global_cutoff_us: np.int64 | None,
    exclude_pages: set[str],
    exclude_auth_values: set[str],
    batch_size: int,
    k: int,
    *,
    verbose: bool = False,
    log_every_batches: int = 10,
) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.int64)

    pf = pq.ParquetFile(parquet_path)
    cols = ["userId", "time", "page", "auth", "__index_level_0__"]
    user_indexer = pd.Index(user_ids)
    exclude_pages_list = list(exclude_pages)
    exclude_auth_list = list(exclude_auth_values)

    # Store top-k (time_us, event_id) per user.
    top: list[list[tuple[int, int]]] = [[] for _ in range(len(user_ids))]

    batches_seen = 0
    rows_seen = 0
    rows_used = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols, use_threads=True):
        batches_seen += 1
        rows_seen += batch.num_rows
        user_arr = batch.column("userId").to_numpy(zero_copy_only=False)
        time_arr = batch.column("time").to_numpy(zero_copy_only=False)
        page_arr = batch.column("page").to_numpy(zero_copy_only=False)
        auth_arr = batch.column("auth").to_numpy(zero_copy_only=False)
        event_id_arr = batch.column("__index_level_0__").to_numpy(zero_copy_only=False).astype(np.int64)

        user_idx = user_indexer.get_indexer(user_arr)
        time_us = time_arr.astype("datetime64[us]").astype("int64")

        mask = user_idx >= 0
        if cutoff_by_user_us is not None:
            mask &= time_us < cutoff_by_user_us[user_idx]
        elif global_cutoff_us is not None:
            mask &= time_us < int(global_cutoff_us)

        if exclude_pages_list:
            mask &= ~np.isin(page_arr, exclude_pages_list)
        if exclude_auth_list:
            mask &= ~np.isin(auth_arr, exclude_auth_list)

        if not mask.any():
            continue

        rows_used += int(np.count_nonzero(mask))

        user_idx = user_idx[mask].astype(np.int32)
        time_us = time_us[mask].astype(np.int64)
        event_id_arr = event_id_arr[mask]

        # Sort by (user_idx, time_us, event_id) so we can take the last k rows per user.
        order = np.lexsort((event_id_arr, time_us, user_idx))
        user_sorted = user_idx[order]
        time_sorted = time_us[order]
        id_sorted = event_id_arr[order]

        uniq_users, first_pos, counts = np.unique(user_sorted, return_index=True, return_counts=True)
        for u, start, cnt in zip(uniq_users.tolist(), first_pos.tolist(), counts.tolist()):
            end = start + cnt
            take = min(k, cnt)
            # last `take` rows for this user
            cand = list(zip(time_sorted[end - take : end].tolist(), id_sorted[end - take : end].tolist()))
            merged = top[u] + cand
            merged.sort()
            top[u] = merged[-k:]

        if verbose and log_every_batches > 0 and (batches_seen % log_every_batches == 0):
            _log(verbose, f"[drop_last_k] batches={batches_seen} rows={rows_seen:,} used={rows_used:,}")

    # Flatten ids to a unique array
    out_ids = [event_id for lst in top for _, event_id in lst]
    return np.array(sorted(set(out_ids)), dtype=np.int64)


def build_user_day_features(
    parquet_path: Path,
    user_ids: list[str],
    registration_by_user: dict[str, pd.Timestamp] | None,
    vocab: dict[str, list[object]],
    calendar_start_day: np.datetime64,
    calendar_num_days: int,
    cutoff_by_user_us: np.ndarray | None,
    global_cutoff_us: np.int64 | None,
    drop_event_ids: np.ndarray,
    exclude_pages: set[str],
    exclude_auth_values: set[str],
    batch_size: int,
    full_calendar: bool,
    *,
    verbose: bool = False,
    log_every_batches: int = 10,
) -> pd.DataFrame:
    pf = pq.ParquetFile(parquet_path)

    pages = [str(x) for x in vocab["pages"]]
    auth_values = [str(x) for x in vocab["auth_values"]]
    methods = [str(x) for x in vocab["methods"]]
    statuses = [int(x) for x in vocab["statuses"]]
    levels = ["free", "paid"]

    page_indexer = pd.Index(pages)
    auth_indexer = pd.Index(auth_values)
    method_indexer = pd.Index(methods)
    status_indexer = pd.Index(statuses)
    user_indexer = pd.Index(user_ids)

    num_users = len(user_ids)
    num_days = int(calendar_num_days)
    num_groups = num_users * num_days

    # Group-level arrays (flattened by group_id = user_idx * num_days + day_idx).
    events_cnt = np.zeros(num_groups, dtype=np.int32)
    sessions_cnt = np.zeros(num_groups, dtype=np.int32)
    nextsong_cnt = np.zeros(num_groups, dtype=np.int32)
    length_sum = np.zeros(num_groups, dtype=np.float32)
    length_sumsq = np.zeros(num_groups, dtype=np.float64)

    # Per user-day time span (for rhythm/gap proxies).
    min_time_us = np.full(num_groups, np.iinfo(np.int64).max, dtype=np.int64)
    max_time_us = np.full(num_groups, -1, dtype=np.int64)

    # Session-level aggregates per user-day (computed within each batch; accumulated across batches).
    # - max events per session
    # - sum/max of session duration seconds
    sess_events_max = np.zeros(num_groups, dtype=np.int32)
    sess_dur_sum_sec = np.zeros(num_groups, dtype=np.float32)
    sess_dur_max_sec = np.zeros(num_groups, dtype=np.float32)

    # Approximate unique song/artist per user-day with 128-bit bitsets (2x uint64).
    # Deterministic via pandas' stable hashing; avoids expensive exact nunique.
    song_bits0 = np.zeros(num_groups, dtype=np.uint64)
    song_bits1 = np.zeros(num_groups, dtype=np.uint64)
    artist_bits0 = np.zeros(num_groups, dtype=np.uint64)
    artist_bits1 = np.zeros(num_groups, dtype=np.uint64)

    first_hour = np.full(num_groups, 99, dtype=np.int16)
    last_hour = np.full(num_groups, -1, dtype=np.int16)
    hour_mask = np.zeros(num_groups, dtype=np.uint32)

    level_free_cnt = np.zeros(num_groups, dtype=np.int32)
    level_paid_cnt = np.zeros(num_groups, dtype=np.int32)
    max_time_free = np.full(num_groups, -1, dtype=np.int64)
    max_time_paid = np.full(num_groups, -1, dtype=np.int64)

    # Categorical count tables stored as 1D arrays (group-major).
    page_cnt = np.zeros(num_groups * len(pages), dtype=np.int32)
    auth_cnt = np.zeros(num_groups * len(auth_values), dtype=np.int32)
    method_cnt = np.zeros(num_groups * len(methods), dtype=np.int32)
    status_cnt = np.zeros(num_groups * len(statuses), dtype=np.int32)

    cols = [
        "userId",
        "time",
        "page",
        "auth",
        "method",
        "status",
        "sessionId",
        "itemInSession",
        "level",
        "length",
        "song",
        "artist",
        "__index_level_0__",
    ]

    drop_event_ids = np.asarray(drop_event_ids, dtype=np.int64)
    has_drop_ids = drop_event_ids.size > 0
    exclude_pages_list = list(exclude_pages)
    exclude_auth_list = list(exclude_auth_values)

    batches_seen = 0
    rows_seen = 0
    rows_used = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols, use_threads=True):
        batches_seen += 1
        rows_seen += batch.num_rows
        user_arr = batch.column("userId").to_numpy(zero_copy_only=False)
        time_arr = batch.column("time").to_numpy(zero_copy_only=False)
        page_arr = batch.column("page").to_numpy(zero_copy_only=False)
        auth_arr = batch.column("auth").to_numpy(zero_copy_only=False)
        method_arr = batch.column("method").to_numpy(zero_copy_only=False)
        status_arr = batch.column("status").to_numpy(zero_copy_only=False)
        session_arr = batch.column("sessionId").to_numpy(zero_copy_only=False)
        item_arr = batch.column("itemInSession").to_numpy(zero_copy_only=False)
        level_arr = batch.column("level").to_numpy(zero_copy_only=False)
        length_arr = batch.column("length").to_numpy(zero_copy_only=False)
        event_id_arr = batch.column("__index_level_0__").to_numpy(zero_copy_only=False).astype(np.int64)

        user_idx = user_indexer.get_indexer(user_arr).astype(np.int32)
        time_us = time_arr.astype("datetime64[us]").astype("int64")

        mask = user_idx >= 0
        if cutoff_by_user_us is not None:
            mask &= time_us < cutoff_by_user_us[user_idx]
        elif global_cutoff_us is not None:
            mask &= time_us < int(global_cutoff_us)

        if exclude_pages_list:
            mask &= ~np.isin(page_arr, exclude_pages_list)
        if exclude_auth_list:
            mask &= ~np.isin(auth_arr, exclude_auth_list)
        if has_drop_ids:
            mask &= ~np.isin(event_id_arr, drop_event_ids)

        # Calendar range filter (compute on the original batch to keep indices aligned for pyarrow.take).
        day_idx_all = (time_arr.astype("datetime64[D]") - calendar_start_day).astype(np.int16)
        mask &= (day_idx_all >= 0) & (day_idx_all < num_days)

        if not mask.any():
            continue

        sel_idx = np.flatnonzero(mask)

        user_idx = user_idx[mask]
        rows_used += int(user_idx.size)
        time_us = time_us[mask]
        time_arr = time_arr[mask]
        page_arr = page_arr[mask]
        auth_arr = auth_arr[mask]
        method_arr = method_arr[mask]
        status_arr = status_arr[mask]
        session_arr = session_arr[mask]
        item_arr = item_arr[mask]
        level_arr = level_arr[mask]
        length_arr = length_arr[mask]
        day_idx = day_idx_all[mask]

        flat_idx = (user_idx.astype(np.int64) * num_days + day_idx.astype(np.int64)).astype(np.int64)

        np.add.at(events_cnt, flat_idx, 1)

        # Per-day time span (min/max timestamp)
        np.minimum.at(min_time_us, flat_idx, time_us)
        np.maximum.at(max_time_us, flat_idx, time_us)

        session_start_mask = item_arr == 0
        if session_start_mask.any():
            np.add.at(sessions_cnt, flat_idx[session_start_mask], 1)

        # Session-level features (events per session + duration per session).
        # Efficient strategy:
        # - group within current batch on (flat_idx, sessionId) using np.unique on a structured key
        # - compute per-session counts and (min,max) time
        # - then reduce back to user-day via np.add.at / np.maximum.at
        # Note: (flat_idx, sessionId) pairs can appear across different batches, but that is fine:
        # - counts and durations are additive/max-safe under batch partitioning.
        try:
            sess_ids = session_arr.astype(np.int64, copy=False)
            if sess_ids.size:
                keys = np.empty(sess_ids.size, dtype=[("g", np.int64), ("s", np.int64)])
                keys["g"] = flat_idx
                keys["s"] = sess_ids
                uniq_keys, inv = np.unique(keys, return_inverse=True)
                n_sess = int(uniq_keys.size)
                if n_sess:
                    sess_cnt = np.bincount(inv, minlength=n_sess).astype(np.int32, copy=False)
                    smin = np.full(n_sess, np.iinfo(np.int64).max, dtype=np.int64)
                    smax = np.full(n_sess, -1, dtype=np.int64)
                    np.minimum.at(smin, inv, time_us)
                    np.maximum.at(smax, inv, time_us)
                    dur_sec = np.maximum(smax - smin, 0).astype(np.float32) * np.float32(1e-6)

                    g = uniq_keys["g"].astype(np.int64, copy=False)
                    np.maximum.at(sess_events_max, g, sess_cnt)
                    np.add.at(sess_dur_sum_sec, g, dur_sec)
                    np.maximum.at(sess_dur_max_sec, g, dur_sec)
        except Exception:
            # Session features are optional; never fail the whole build.
            pass

        time_h = time_arr.astype("datetime64[h]").astype(np.int64)
        hour = (time_h % 24).astype(np.int16)
        hour_bits = (np.uint32(1) << hour.astype(np.uint32))
        np.bitwise_or.at(hour_mask, flat_idx, hour_bits)
        np.minimum.at(first_hour, flat_idx, hour)
        np.maximum.at(last_hour, flat_idx, hour)

        # Categorical counts
        page_code = page_indexer.get_indexer(page_arr).astype(np.int16)
        valid = page_code >= 0
        if valid.any():
            combined = flat_idx[valid] * len(pages) + page_code[valid].astype(np.int64)
            np.add.at(page_cnt, combined, 1)

        auth_code = auth_indexer.get_indexer(auth_arr).astype(np.int16)
        valid = auth_code >= 0
        if valid.any():
            combined = flat_idx[valid] * len(auth_values) + auth_code[valid].astype(np.int64)
            np.add.at(auth_cnt, combined, 1)

        method_code = method_indexer.get_indexer(method_arr).astype(np.int16)
        valid = method_code >= 0
        if valid.any():
            combined = flat_idx[valid] * len(methods) + method_code[valid].astype(np.int64)
            np.add.at(method_cnt, combined, 1)

        status_code = status_indexer.get_indexer(status_arr).astype(np.int16)
        valid = status_code >= 0
        if valid.any():
            combined = flat_idx[valid] * len(statuses) + status_code[valid].astype(np.int64)
            np.add.at(status_cnt, combined, 1)

        # NextSong length features
        nextsong_mask = page_arr == "NextSong"
        if nextsong_mask.any():
            flat_ns = flat_idx[nextsong_mask]
            np.add.at(nextsong_cnt, flat_ns, 1)
            ns_len = length_arr[nextsong_mask]
            # Filter null/NaN lengths
            if ns_len.dtype.kind in {"f"}:
                valid_len = ~np.isnan(ns_len)
            else:
                valid_len = ns_len != None  # noqa: E711
            if valid_len.any():
                flat_ns2 = flat_ns[valid_len]
                v = ns_len[valid_len].astype(np.float64)
                np.add.at(length_sum, flat_ns2, v.astype(np.float32))
                np.add.at(length_sumsq, flat_ns2, v * v)

            # Approximate unique song/artist counts
            idx_ns = sel_idx[nextsong_mask].astype(np.int32, copy=False)

            ns_song = batch.column("song").take(pa.array(idx_ns)).to_numpy(zero_copy_only=False)
            valid_song = pd.notna(ns_song)
            if valid_song.any():
                song_hash = pd.util.hash_pandas_object(pd.Series(ns_song[valid_song], copy=False), index=False).values
                song_bucket = (song_hash.astype(np.uint64) & np.uint64(127)).astype(np.uint8)
                song_word = (song_bucket >> 6).astype(np.uint8)
                song_bit = (np.uint64(1) << (song_bucket & np.uint8(63))).astype(np.uint64)
                flat_song = flat_ns[valid_song]
                w0 = song_word == 0
                if w0.any():
                    np.bitwise_or.at(song_bits0, flat_song[w0], song_bit[w0])
                w1 = ~w0
                if w1.any():
                    np.bitwise_or.at(song_bits1, flat_song[w1], song_bit[w1])

            ns_artist = batch.column("artist").take(pa.array(idx_ns)).to_numpy(zero_copy_only=False)
            valid_artist = pd.notna(ns_artist)
            if valid_artist.any():
                artist_hash = pd.util.hash_pandas_object(pd.Series(ns_artist[valid_artist], copy=False), index=False).values
                artist_bucket = (artist_hash.astype(np.uint64) & np.uint64(127)).astype(np.uint8)
                artist_word = (artist_bucket >> 6).astype(np.uint8)
                artist_bit = (np.uint64(1) << (artist_bucket & np.uint8(63))).astype(np.uint64)
                flat_artist = flat_ns[valid_artist]
                w0 = artist_word == 0
                if w0.any():
                    np.bitwise_or.at(artist_bits0, flat_artist[w0], artist_bit[w0])
                w1 = ~w0
                if w1.any():
                    np.bitwise_or.at(artist_bits1, flat_artist[w1], artist_bit[w1])

        # Level features (free/paid)
        free_mask = level_arr == "free"
        paid_mask = level_arr == "paid"
        if free_mask.any():
            flat_free = flat_idx[free_mask]
            np.add.at(level_free_cnt, flat_free, 1)
            np.maximum.at(max_time_free, flat_free, time_us[free_mask])
        if paid_mask.any():
            flat_paid = flat_idx[paid_mask]
            np.add.at(level_paid_cnt, flat_paid, 1)
            np.maximum.at(max_time_paid, flat_paid, time_us[paid_mask])

        if verbose and log_every_batches > 0 and (batches_seen % log_every_batches == 0):
            _log(verbose, f"[user_day] batches={batches_seen} rows={rows_seen:,} used={rows_used:,}")

    # Reshape categorical counts
    page_cnt_m = page_cnt.reshape(num_groups, len(pages))
    auth_cnt_m = auth_cnt.reshape(num_groups, len(auth_values))
    method_cnt_m = method_cnt.reshape(num_groups, len(methods))
    status_cnt_m = status_cnt.reshape(num_groups, len(statuses))

    pages_unique = (page_cnt_m > 0).sum(axis=1).astype(np.int16)

    # Bitcount for active hours (uint32 -> 4 bytes)
    byte_popcnt = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    hours_active = byte_popcnt[hour_mask.view(np.uint8).reshape(-1, 4)].sum(axis=1).astype(np.int16)

    song_bucket_unique = byte_popcnt[song_bits0.view(np.uint8).reshape(-1, 8)].sum(axis=1).astype(np.int16)
    song_bucket_unique += byte_popcnt[song_bits1.view(np.uint8).reshape(-1, 8)].sum(axis=1).astype(np.int16)
    artist_bucket_unique = byte_popcnt[artist_bits0.view(np.uint8).reshape(-1, 8)].sum(axis=1).astype(np.int16)
    artist_bucket_unique += byte_popcnt[artist_bits1.view(np.uint8).reshape(-1, 8)].sum(axis=1).astype(np.int16)

    # Time-based features post-process
    first_hour_out = first_hour.copy()
    first_hour_out[events_cnt == 0] = -1
    last_hour_out = last_hour.copy()

    # Day span in seconds (0 for inactive days)
    span_sec = np.zeros(num_groups, dtype=np.float32)
    has_events = events_cnt > 0
    if has_events.any():
        span_us = (max_time_us[has_events] - min_time_us[has_events]).astype(np.int64)
        span_us = np.maximum(span_us, 0)
        span_sec[has_events] = span_us.astype(np.float32) * np.float32(1e-6)

    # NextSong summary stats
    listen_sum = length_sum.astype(np.float32)
    listen_mean = np.zeros(num_groups, dtype=np.float32)
    listen_std = np.zeros(num_groups, dtype=np.float32)
    nonzero = nextsong_cnt > 0
    if nonzero.any():
        c = nextsong_cnt[nonzero].astype(np.float64)
        s = length_sum[nonzero].astype(np.float64)
        ss = length_sumsq[nonzero]
        mean = s / c
        var = np.maximum(ss / c - mean * mean, 0.0)
        listen_mean[nonzero] = mean.astype(np.float32)
        listen_std[nonzero] = np.sqrt(var).astype(np.float32)

    # Level derived
    paid_ratio = np.zeros(num_groups, dtype=np.float32)
    paid_ratio[has_events] = (level_paid_cnt[has_events] / events_cnt[has_events]).astype(np.float32)
    level_changed = ((level_paid_cnt > 0) & (level_free_cnt > 0)).astype(np.int8)
    level_last_paid = (max_time_paid > max_time_free).astype(np.int8)
    level_last_paid[(max_time_paid < 0) & (max_time_free < 0)] = 0

    # Derived rates using common pages/statuses/methods if available
    page_to_col = {p: i for i, p in enumerate(pages)}
    status_to_col = {s: i for i, s in enumerate(statuses)}
    method_to_col = {m: i for i, m in enumerate(methods)}

    def _safe_get_page_cnt(name: str) -> np.ndarray:
        idx = page_to_col.get(name)
        if idx is None:
            return np.zeros(num_groups, dtype=np.int32)
        return page_cnt_m[:, idx]

    thumbs_up_cnt = _safe_get_page_cnt("Thumbs Up")
    thumbs_down_cnt = _safe_get_page_cnt("Thumbs Down")
    roll_advert_cnt = _safe_get_page_cnt("Roll Advert")
    add_friend_cnt = _safe_get_page_cnt("Add Friend")
    error_page_cnt = _safe_get_page_cnt("Error")

    denom_nextsong = np.maximum(nextsong_cnt.astype(np.float32), 1.0)
    thumbs_up_rate = (thumbs_up_cnt.astype(np.float32) / denom_nextsong).astype(np.float32)
    thumbs_down_rate = (thumbs_down_cnt.astype(np.float32) / denom_nextsong).astype(np.float32)
    ad_rate = (roll_advert_cnt.astype(np.float32) / denom_nextsong).astype(np.float32)

    has_ns = nextsong_cnt > 0
    song_repeat_rate = np.zeros(num_groups, dtype=np.float32)
    artist_repeat_rate = np.zeros(num_groups, dtype=np.float32)
    song_repeat_rate[has_ns] = (1.0 - (song_bucket_unique[has_ns].astype(np.float32) / denom_nextsong[has_ns])).astype(
        np.float32
    )
    artist_repeat_rate[has_ns] = (
        1.0 - (artist_bucket_unique[has_ns].astype(np.float32) / denom_nextsong[has_ns])
    ).astype(np.float32)

    denom_sessions = np.maximum(sessions_cnt.astype(np.float32), 1.0)
    add_friend_per_session = (add_friend_cnt.astype(np.float32) / denom_sessions).astype(np.float32)

    status_404_cnt = (
        status_cnt_m[:, status_to_col[404]] if 404 in status_to_col else np.zeros(num_groups, dtype=np.int32)
    )
    error_rate = np.zeros(num_groups, dtype=np.float32)
    error_rate[has_events] = (
        (status_404_cnt[has_events] + error_page_cnt[has_events]).astype(np.float32) / events_cnt[has_events]
    ).astype(np.float32)

    get_cnt = method_cnt_m[:, method_to_col["GET"]] if "GET" in method_to_col else np.zeros(num_groups, dtype=np.int32)
    put_cnt = method_cnt_m[:, method_to_col["PUT"]] if "PUT" in method_to_col else np.zeros(num_groups, dtype=np.int32)
    get_rate = np.zeros(num_groups, dtype=np.float32)
    get_rate[has_events] = (get_cnt[has_events].astype(np.float32) / events_cnt[has_events]).astype(np.float32)

    # Session derived (mean duration per session)
    sess_dur_mean_sec = np.zeros(num_groups, dtype=np.float32)
    denom_sessions_f = np.maximum(sessions_cnt.astype(np.float32), 1.0)
    sess_dur_mean_sec = (sess_dur_sum_sec / denom_sessions_f).astype(np.float32)

    # Rhythm / intensity proxies
    denom_events_minus1 = np.maximum(events_cnt.astype(np.float32) - 1.0, 1.0)
    mean_gap_proxy_sec = np.zeros(num_groups, dtype=np.float32)
    mean_gap_proxy_sec[has_events] = (span_sec[has_events] / denom_events_minus1[has_events]).astype(np.float32)

    denom_active_hours = np.maximum(hours_active.astype(np.float32), 1.0)
    events_per_active_hour = np.zeros(num_groups, dtype=np.float32)
    sessions_per_active_hour = np.zeros(num_groups, dtype=np.float32)
    nextsong_per_active_hour = np.zeros(num_groups, dtype=np.float32)
    events_per_active_hour[has_events] = (events_cnt[has_events].astype(np.float32) / denom_active_hours[has_events]).astype(
        np.float32
    )
    sessions_per_active_hour[has_events] = (
        sessions_cnt[has_events].astype(np.float32) / denom_active_hours[has_events]
    ).astype(np.float32)
    nextsong_per_active_hour[has_events] = (
        nextsong_cnt[has_events].astype(np.float32) / denom_active_hours[has_events]
    ).astype(np.float32)

    inactive_hours = (24 - hours_active).astype(np.int16)

    # Diversity / concentration features (cheap from count matrices)
    def _entropy_simpson_top1(counts: np.ndarray, total: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_f = np.maximum(total.astype(np.float32), 1.0)
        p = counts.astype(np.float32) / total_f[:, None]
        # entropy: -sum p log p
        eps = np.float32(1e-12)
        ent = -(p * np.log(p + eps)).sum(axis=1, dtype=np.float32)
        k = int(counts.shape[1])
        if k > 1:
            ent_norm = (ent / np.float32(np.log(float(k)))).astype(np.float32)
        else:
            ent_norm = np.zeros_like(ent, dtype=np.float32)
        simpson = (np.float32(1.0) - (p * p).sum(axis=1, dtype=np.float32)).astype(np.float32)
        top1 = (counts.max(axis=1).astype(np.float32) / total_f).astype(np.float32)
        # zero out for inactive days for stability
        ent_norm[~has_events] = 0.0
        simpson[~has_events] = 0.0
        top1[~has_events] = 0.0
        return ent_norm, simpson, top1

    page_entropy_norm, page_simpson, page_top1_share = _entropy_simpson_top1(page_cnt_m, events_cnt)
    status_entropy_norm, status_simpson, status_top1_share = _entropy_simpson_top1(status_cnt_m, events_cnt)
    method_entropy_norm, method_simpson, method_top1_share = _entropy_simpson_top1(method_cnt_m, events_cnt)
    auth_entropy_norm, auth_simpson, auth_top1_share = _entropy_simpson_top1(auth_cnt_m, events_cnt)

    # Build output DataFrame
    day_values = (calendar_start_day + np.arange(num_days).astype("timedelta64[D]")).astype("datetime64[ns]")
    out_user = np.repeat(np.asarray(user_ids, dtype=object), num_days)
    out_day = np.tile(day_values, num_users)

    data: dict[str, object] = {
        "userId": out_user,
        "day": out_day,
        "d_events": events_cnt,
        "d_sessions": sessions_cnt,
        "d_pages_unique": pages_unique,
        "d_active_hours": hours_active,
        "d_first_hour": first_hour_out.astype(np.int16),
        "d_last_hour": last_hour_out.astype(np.int16),
        "d_span_sec": span_sec,
        "d_mean_gap_proxy_sec": mean_gap_proxy_sec,
        "d_inactive_hours": inactive_hours,
        "d_nextsong_cnt": nextsong_cnt,
        "d_listen_sec_sum": listen_sum,
        "d_listen_sec_mean": listen_mean,
        "d_listen_sec_std": listen_std,
        "d_song_bucket_unique": song_bucket_unique,
        "d_artist_bucket_unique": artist_bucket_unique,
        "d_song_repeat_rate": song_repeat_rate,
        "d_artist_repeat_rate": artist_repeat_rate,
        "d_level_free_cnt": level_free_cnt,
        "d_level_paid_cnt": level_paid_cnt,
        "d_paid_ratio": paid_ratio,
        "d_level_last_paid": level_last_paid,
        "d_level_changed_in_day": level_changed,
        "d_get_rate": get_rate,
        "d_error_rate": error_rate,
        "d_thumbs_up_rate": thumbs_up_rate,
        "d_thumbs_down_rate": thumbs_down_rate,
        "d_ad_rate": ad_rate,
        "d_add_friend_per_session": add_friend_per_session,
        # Session structure signals
        "d_sess_events_max": sess_events_max,
        "d_sess_dur_sec_sum": sess_dur_sum_sec,
        "d_sess_dur_sec_mean": sess_dur_mean_sec,
        "d_sess_dur_sec_max": sess_dur_max_sec,
        # Intensity per active hour
        "d_events_per_active_hour": events_per_active_hour,
        "d_sessions_per_active_hour": sessions_per_active_hour,
        "d_nextsong_per_active_hour": nextsong_per_active_hour,
        # Diversity/concentration
        "d_page_entropy_norm": page_entropy_norm,
        "d_page_simpson": page_simpson,
        "d_page_top1_share": page_top1_share,
        "d_status_entropy_norm": status_entropy_norm,
        "d_status_simpson": status_simpson,
        "d_status_top1_share": status_top1_share,
        "d_method_entropy_norm": method_entropy_norm,
        "d_method_simpson": method_simpson,
        "d_method_top1_share": method_top1_share,
        "d_auth_entropy_norm": auth_entropy_norm,
        "d_auth_simpson": auth_simpson,
        "d_auth_top1_share": auth_top1_share,
    }

    # Add count columns (page/method/status/auth)
    for i, p in enumerate(pages):
        data[f"d_page_{_slugify(p)}_cnt"] = page_cnt_m[:, i]
    for i, a in enumerate(auth_values):
        data[f"d_auth_{_slugify(a)}_cnt"] = auth_cnt_m[:, i]
    for i, m in enumerate(methods):
        data[f"d_method_{_slugify(m)}_cnt"] = method_cnt_m[:, i]
    for i, s in enumerate(statuses):
        data[f"d_status_{s}_cnt"] = status_cnt_m[:, i]

    # Static-derived: tenure days since registration (per day).
    if registration_by_user is not None:
        reg_days = np.array(
            [
                np.datetime64(pd.Timestamp(registration_by_user[uid]).normalize().to_datetime64(), "D")
                if uid in registration_by_user
                else np.datetime64("NaT")
                for uid in user_ids
            ],
            dtype="datetime64[D]",
        )
        out_reg = np.repeat(reg_days, num_days)
        out_day_d = out_day.astype("datetime64[D]")
        tenure = np.full(out_day.shape[0], -1, dtype=np.int16)
        valid = ~np.isnat(out_reg)
        if valid.any():
            delta = (out_day_d[valid] - out_reg[valid]).astype("timedelta64[D]").astype(np.int32)
            delta = np.maximum(delta, 0)
            tenure[valid] = np.minimum(delta, 32767).astype(np.int16)
        data["u_tenure_days"] = tenure

    # Cross-day rolling/trend/state features (computed on dense full calendar).
    # Shape: (num_users, num_days)
    events_2d = events_cnt.reshape(num_users, num_days)
    nextsong_2d = nextsong_cnt.reshape(num_users, num_days)
    pages_unique_2d = pages_unique.reshape(num_users, num_days).astype(np.float32, copy=False)
    paid_ratio_2d = paid_ratio.reshape(num_users, num_days).astype(np.float32, copy=False)
    error_rate_2d = error_rate.reshape(num_users, num_days).astype(np.float32, copy=False)
    listen_mean_2d = listen_mean.reshape(num_users, num_days).astype(np.float32, copy=False)

    level_last_paid_2d = level_last_paid.reshape(num_users, num_days).astype(np.int8, copy=False)

    def _rolling_sum_2d(x: np.ndarray, window: int) -> np.ndarray:
        w = int(window)
        if w <= 1:
            return x.astype(np.float32, copy=False)
        cs = np.cumsum(x.astype(np.float32, copy=False), axis=1)
        out = cs.copy()
        out[:, w:] = cs[:, w:] - cs[:, :-w]
        return out

    def _rolling_mean_2d_fixed_window(x: np.ndarray, window: int) -> np.ndarray:
        s = _rolling_sum_2d(x, window)
        return (s / float(window)).astype(np.float32)

    def _rolling_std_2d_fixed_window(x: np.ndarray, window: int) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        s = _rolling_sum_2d(x, window)
        ss = _rolling_sum_2d(x * x, window)
        mean = s / float(window)
        ex2 = ss / float(window)
        var = np.maximum(ex2 - mean * mean, 0.0)
        return np.sqrt(var, dtype=np.float32)

    # Multi-scale rolling features (3/7/14/28) for key signals.
    active_2d = (events_2d > 0).astype(np.int16)
    active_f = active_2d.astype(np.float32)
    paid_active_2d = (level_paid_cnt.reshape(num_users, num_days) > 0).astype(np.int16)
    paid_active_f = paid_active_2d.astype(np.float32)

    # Level switch dynamics (only count switches between days that both have activity).
    # level_state: -1 for inactive days, else {0,1} from d_level_last_paid
    level_state_2d = np.where(active_2d > 0, level_last_paid_2d.astype(np.int16), np.int16(-1)).astype(np.int16)
    level_switch_2d = np.zeros((num_users, num_days), dtype=np.float32)
    if num_days >= 2:
        prev = level_state_2d[:, :-1]
        cur = level_state_2d[:, 1:]
        sw = (prev >= 0) & (cur >= 0) & (prev != cur)
        level_switch_2d[:, 1:] = sw.astype(np.float32)

    for w in (3, 7, 14, 28):
        roll_events = _rolling_sum_2d(events_2d, w)
        roll_nextsong = _rolling_sum_2d(nextsong_2d, w)

        data[f"x_roll{w}_events_sum"] = roll_events.reshape(-1).astype(np.float32)
        data[f"x_roll{w}_nextsong_sum"] = roll_nextsong.reshape(-1).astype(np.float32)
        data[f"x_roll{w}_events_mean"] = (roll_events / float(w)).reshape(-1).astype(np.float32)
        data[f"x_roll{w}_nextsong_mean"] = (roll_nextsong / float(w)).reshape(-1).astype(np.float32)

        data[f"x_roll{w}_active_days"] = _rolling_sum_2d(active_f, w).reshape(-1).astype(np.float32)
        data[f"x_roll{w}_paid_ratio_mean"] = _rolling_mean_2d_fixed_window(paid_ratio_2d, w).reshape(-1).astype(np.float32)
        data[f"x_roll{w}_error_rate_mean"] = _rolling_mean_2d_fixed_window(error_rate_2d, w).reshape(-1).astype(np.float32)
        data[f"x_roll{w}_pages_unique_mean"] = _rolling_mean_2d_fixed_window(pages_unique_2d, w).reshape(-1).astype(np.float32)

        # Subscription instability: how often level switches in last w days
        data[f"x_roll{w}_level_switch_cnt"] = _rolling_sum_2d(level_switch_2d, w).reshape(-1).astype(np.float32)

        if w in (7, 14):
            data[f"x_roll{w}_paid_days"] = _rolling_sum_2d(paid_active_f, w).reshape(-1).astype(np.float32)

    # Keep legacy roll7 names (same semantics; computed above).
    roll7_events = _rolling_sum_2d(events_2d, 7)
    roll7_nextsong = _rolling_sum_2d(nextsong_2d, 7)
    data["x_roll7_events_sum"] = roll7_events.reshape(-1).astype(np.float32)
    data["x_roll7_nextsong_sum"] = roll7_nextsong.reshape(-1).astype(np.float32)
    data["x_roll7_events_mean"] = (roll7_events / 7.0).reshape(-1).astype(np.float32)
    data["x_roll7_nextsong_mean"] = (roll7_nextsong / 7.0).reshape(-1).astype(np.float32)

    # last7 mean - prev7 mean (needs 14d context)
    trend7_events = np.zeros_like(roll7_events, dtype=np.float32)
    trend7_nextsong = np.zeros_like(roll7_nextsong, dtype=np.float32)
    if num_days >= 14:
        roll14_events = _rolling_sum_2d(events_2d, 14)
        roll14_nextsong = _rolling_sum_2d(nextsong_2d, 14)
        prev7_events = np.maximum(roll14_events - roll7_events, 0.0)
        prev7_nextsong = np.maximum(roll14_nextsong - roll7_nextsong, 0.0)
        trend7_events = (roll7_events - prev7_events) / 7.0
        trend7_nextsong = (roll7_nextsong - prev7_nextsong) / 7.0
    data["x_trend7_events"] = trend7_events.reshape(-1).astype(np.float32)
    data["x_trend7_nextsong"] = trend7_nextsong.reshape(-1).astype(np.float32)

    # Active-day count in last 7 days
    roll7_active = _rolling_sum_2d(active_f, 7)
    data["x_roll7_active_days"] = roll7_active.reshape(-1).astype(np.float32)

    # Days since last active (0 on active days, -1 before first activity)
    last_idx = np.full(num_users, -1, dtype=np.int16)
    d_since = np.full((num_users, num_days), -1, dtype=np.int16)
    for t in range(num_days):
        a = active_2d[:, t] > 0
        last_idx[a] = t
        has = last_idx >= 0
        d_since[has, t] = (t - last_idx[has]).astype(np.int16)
    data["x_days_since_last_active"] = d_since.reshape(-1)

    # Paid temporal features: days since last paid (0 on paid day, -1 before first paid)
    last_paid_idx = np.full(num_users, -1, dtype=np.int16)
    d_since_paid = np.full((num_users, num_days), -1, dtype=np.int16)
    for t in range(num_days):
        p = paid_active_2d[:, t] > 0
        last_paid_idx[p] = t
        has = last_paid_idx >= 0
        d_since_paid[has, t] = (t - last_paid_idx[has]).astype(np.int16)
    data["x_days_since_last_paid"] = d_since_paid.reshape(-1)

    paid_streak = np.zeros((num_users, num_days), dtype=np.int16)
    for t in range(num_days):
        if t == 0:
            paid_streak[:, t] = (paid_active_2d[:, t] > 0).astype(np.int16)
        else:
            paid_streak[:, t] = np.where(paid_active_2d[:, t] > 0, paid_streak[:, t - 1] + 1, 0).astype(np.int16)
    data["x_paid_streak"] = paid_streak.reshape(-1)

    # Days since last level switch (0 on switch day, -1 before first switch)
    last_sw_idx = np.full(num_users, -1, dtype=np.int16)
    d_since_sw = np.full((num_users, num_days), -1, dtype=np.int16)
    for t in range(num_days):
        sw = level_switch_2d[:, t] > 0
        last_sw_idx[sw] = t
        has = last_sw_idx >= 0
        d_since_sw[has, t] = (t - last_sw_idx[has]).astype(np.int16)
    data["x_days_since_last_level_switch"] = d_since_sw.reshape(-1)

    # Active streak length (consecutive active days)
    streak = np.zeros((num_users, num_days), dtype=np.int16)
    for t in range(num_days):
        if t == 0:
            streak[:, t] = (active_2d[:, t] > 0).astype(np.int16)
        else:
            streak[:, t] = np.where(active_2d[:, t] > 0, streak[:, t - 1] + 1, 0).astype(np.int16)
    data["x_active_streak"] = streak.reshape(-1)

    # Intensity (daily)
    denom_sessions = np.maximum(sessions_cnt.astype(np.float32), 1.0)
    data["d_events_per_session"] = (events_cnt.astype(np.float32) / denom_sessions).astype(np.float32)
    data["d_nextsong_per_session"] = (nextsong_cnt.astype(np.float32) / denom_sessions).astype(np.float32)

    # Volatility (rolling std)
    data["x_roll7_std_events"] = _rolling_std_2d_fixed_window(events_2d.astype(np.float32, copy=False), 7).reshape(-1).astype(
        np.float32
    )
    data["x_roll7_std_listen_sec_mean"] = _rolling_std_2d_fixed_window(listen_mean_2d, 7).reshape(-1).astype(np.float32)

    # Calendar: day of week (0=Mon..6=Sun)
    try:
        data["d_day_of_week"] = pd.DatetimeIndex(out_day).dayofweek.astype(np.int8)
    except Exception:
        data["d_day_of_week"] = np.zeros(out_day.shape[0], dtype=np.int8)

    # Relative-change features
    eps = np.float32(1e-3)
    roll7_events_mean = (roll7_events / 7.0).astype(np.float32)
    roll7_nextsong_mean = (roll7_nextsong / 7.0).astype(np.float32)
    data["x_today_over_roll7_events_mean"] = (
        events_2d.astype(np.float32) / np.maximum(roll7_events_mean, eps)
    ).reshape(-1).astype(np.float32)
    data["x_today_over_roll7_nextsong_mean"] = (
        nextsong_2d.astype(np.float32) / np.maximum(roll7_nextsong_mean, eps)
    ).reshape(-1).astype(np.float32)

    roll28_events_mean = (_rolling_sum_2d(events_2d, 28) / 28.0).astype(np.float32)
    roll28_nextsong_mean = (_rolling_sum_2d(nextsong_2d, 28) / 28.0).astype(np.float32)
    data["x_roll7_minus_roll28_events_mean"] = (roll7_events_mean - roll28_events_mean).reshape(-1).astype(np.float32)
    data["x_roll7_minus_roll28_nextsong_mean"] = (roll7_nextsong_mean - roll28_nextsong_mean).reshape(-1).astype(
        np.float32
    )

    df = pd.DataFrame(data)
    if not full_calendar:
        df = df[df["d_events"] > 0].reset_index(drop=True)
    return df


def split_users(
    user_ids: list[str],
    split_cfg: SplitConfig,
) -> tuple[list[str], list[str]]:
    low = float(split_cfg.val_prob_low)
    high = float(split_cfg.val_prob_high)
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"val_prob range must satisfy 0<=low<high<=1, got {low},{high}")

    val_users: list[str] = []
    train_users: list[str] = []
    for uid in user_ids:
        p = _hash_to_unit_interval(f"{split_cfg.split_seed}:{uid}")
        if low <= p < high:
            val_users.append(uid)
        else:
            train_users.append(uid)
    return train_users, val_users


def build_and_cache_features(
    paths: DatasetPaths,
    feature_cfg: FeatureBuildConfig,
    cache_dir: Path,
    force_recompute: bool = False,
    *,
    verbose: bool = True,
) -> BuildArtifacts:
    feature_cfg = feature_cfg.normalized()
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_fp = _file_fingerprint(paths.train_path)
    test_fp = _file_fingerprint(paths.test_path)

    feature_payload = asdict(feature_cfg)
    feature_payload.pop("batch_size", None)

    cache_key = _stable_hash(
        {
            "feature_cfg": feature_payload,
            "train_fp": train_fp,
            "test_fp": test_fp,
            # Bump when feature schema changes.
            "version": 4,
        }
    )
    run_dir = cache_dir / cache_key
    meta_path = run_dir / "meta.json"
    artifacts_path = run_dir / "artifacts.json"

    train_features_path = run_dir / "train_user_day.parquet"
    train_labels_path = run_dir / "train_labels.parquet"
    test_features_path = run_dir / "test_user_day.parquet"

    _log(verbose, f"[build] cache_key={cache_key} cache_dir={run_dir}")

    if (
        (not force_recompute)
        and artifacts_path.exists()
        and train_features_path.exists()
        and train_labels_path.exists()
        and test_features_path.exists()
    ):
        _log(verbose, "[build] cache hit, skipping feature recompute")
        meta = json.loads(artifacts_path.read_text(encoding="utf-8"))
        return BuildArtifacts(
            vocab=meta["vocab"],
            calendar_start_day=meta["calendar_start_day"],
            calendar_num_days=int(meta["calendar_num_days"]),
            train_features_path=train_features_path,
            train_labels_path=train_labels_path,
            test_features_path=test_features_path,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "feature_cfg": asdict(feature_cfg),
                "train_fp": train_fp,
                "test_fp": test_fp,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # 1) Scan metadata (also gives us global calendar range).
    _log(verbose, "[build] scanning train metadata...")
    (
        train_user_ids,
        train_end_time_by_user,
        churn_time_by_user,
        train_global_max_time,
        train_registration_by_user,
        train_pages,
        train_auth,
        train_methods,
        train_statuses,
    ) = scan_train_metadata(
        paths.train_path,
        batch_size=feature_cfg.batch_size,
        verbose=verbose,
        log_every_batches=5,
    )
    _log(
        verbose,
        f"[build] train users={len(train_user_ids):,} churn_users={len(churn_time_by_user):,} max_time={train_global_max_time}",
    )

    _log(verbose, "[build] scanning test metadata...")
    (
        test_user_ids,
        test_end_time_by_user,
        test_global_max_time,
        test_registration_by_user,
        test_pages,
        test_auth,
        test_methods,
        test_statuses,
    ) = scan_test_metadata(
        paths.test_path,
        batch_size=feature_cfg.batch_size,
        verbose=verbose,
        log_every_batches=5,
    )
    _log(verbose, f"[build] test users={len(test_user_ids):,} max_time={test_global_max_time}")

    global_max = max(train_global_max_time, test_global_max_time)
    global_min = pd.Timestamp("2018-10-01")
    calendar_start_day, calendar_num_days = _build_calendar(global_min, global_max)
    _log(verbose, f"[build] calendar start={pd.to_datetime(calendar_start_day).date()} days={calendar_num_days}")

    exclude_pages = set(feature_cfg.exclude_pages)
    exclude_auth = set(feature_cfg.exclude_auth_values)

    vocab = {
        "pages": sorted((train_pages | test_pages) - exclude_pages),
        "auth_values": sorted((train_auth | test_auth) - exclude_auth),
        "methods": sorted(train_methods | test_methods),
        "statuses": sorted(train_statuses | test_statuses),
    }
    _log(
        verbose,
        f"[build] vocab pages={len(vocab['pages'])} auth={len(vocab['auth_values'])} methods={len(vocab['methods'])} statuses={len(vocab['statuses'])}",
    )

    # 2) Compute cutoffs + labels
    _log(
        verbose,
        f"[build] cutoff_time={feature_cfg.cutoff_time} horizon_days={feature_cfg.horizon_days} label_mode={feature_cfg.label_mode}",
    )
    train_cutoff_by_user_us, train_global_cutoff_us = _compute_cutoff_by_user_us(
        train_user_ids, train_end_time_by_user, feature_cfg.cutoff_time
    )
    test_cutoff_by_user_us, test_global_cutoff_us = _compute_cutoff_by_user_us(
        test_user_ids, test_end_time_by_user, feature_cfg.cutoff_time
    )

    # Global cutoff sanity check (needs label window inside data range).
    if (feature_cfg.label_mode == "horizon") and isinstance(feature_cfg.cutoff_time, str):
        cutoff_ts = _to_timestamp(feature_cfg.cutoff_time)
        horizon_end = cutoff_ts + pd.Timedelta(days=feature_cfg.horizon_days)
        if horizon_end > global_max:
            raise ValueError(
                f"cutoff_time={feature_cfg.cutoff_time} + horizon_days={feature_cfg.horizon_days} exceeds data max {global_max}"
            )

    _log(verbose, "[build] building labels...")
    labels_df = build_train_labels(
        train_user_ids=train_user_ids,
        churn_time_by_user=churn_time_by_user,
        cutoff_by_user_us=train_cutoff_by_user_us,
        global_cutoff_us=train_global_cutoff_us,
        horizon_days=feature_cfg.horizon_days,
        label_mode=feature_cfg.label_mode,
    )
    _log(verbose, f"[build] labels positive={int(labels_df['label'].sum()):,} / {len(labels_df):,}")

    # 3) Optional: compute drop-last-k event ids (train + test separately, aligned to their own cutoff windows)
    if feature_cfg.drop_last_k_events > 0:
        _log(verbose, f"[build] computing drop_last_k_events={feature_cfg.drop_last_k_events} (train)...")
    train_drop_ids = compute_drop_last_k_event_ids(
        parquet_path=paths.train_path,
        user_ids=train_user_ids,
        cutoff_by_user_us=train_cutoff_by_user_us,
        global_cutoff_us=train_global_cutoff_us,
        exclude_pages=exclude_pages,
        exclude_auth_values=exclude_auth,
        batch_size=feature_cfg.batch_size,
        k=feature_cfg.drop_last_k_events,
        verbose=verbose and (feature_cfg.drop_last_k_events > 0),
        log_every_batches=10,
    )
    if feature_cfg.drop_last_k_events > 0:
        _log(verbose, f"[build] train dropped event ids: {train_drop_ids.size:,}")

    if feature_cfg.drop_last_k_events > 0:
        _log(verbose, f"[build] computing drop_last_k_events={feature_cfg.drop_last_k_events} (test)...")
    test_drop_ids = compute_drop_last_k_event_ids(
        parquet_path=paths.test_path,
        user_ids=test_user_ids,
        cutoff_by_user_us=test_cutoff_by_user_us,
        global_cutoff_us=test_global_cutoff_us,
        exclude_pages=exclude_pages,
        exclude_auth_values=exclude_auth,
        batch_size=feature_cfg.batch_size,
        k=feature_cfg.drop_last_k_events,
        verbose=verbose and (feature_cfg.drop_last_k_events > 0),
        log_every_batches=10,
    )
    if feature_cfg.drop_last_k_events > 0:
        _log(verbose, f"[build] test dropped event ids: {test_drop_ids.size:,}")

    # 4) Build daily features
    _log(verbose, "[build] building train user-day features...")
    train_features_df = build_user_day_features(
        parquet_path=paths.train_path,
        user_ids=train_user_ids,
        registration_by_user=train_registration_by_user,
        vocab=vocab,
        calendar_start_day=calendar_start_day,
        calendar_num_days=calendar_num_days,
        cutoff_by_user_us=train_cutoff_by_user_us,
        global_cutoff_us=train_global_cutoff_us,
        drop_event_ids=train_drop_ids,
        exclude_pages=exclude_pages,
        exclude_auth_values=exclude_auth,
        batch_size=feature_cfg.batch_size,
        full_calendar=feature_cfg.full_calendar,
        verbose=verbose,
        log_every_batches=10,
    )
    _log(verbose, f"[build] train user-day shape={train_features_df.shape}")

    _log(verbose, "[build] building test user-day features...")
    test_features_df = build_user_day_features(
        parquet_path=paths.test_path,
        user_ids=test_user_ids,
        registration_by_user=test_registration_by_user,
        vocab=vocab,
        calendar_start_day=calendar_start_day,
        calendar_num_days=calendar_num_days,
        cutoff_by_user_us=test_cutoff_by_user_us,
        global_cutoff_us=test_global_cutoff_us,
        drop_event_ids=test_drop_ids,
        exclude_pages=exclude_pages,
        exclude_auth_values=exclude_auth,
        batch_size=feature_cfg.batch_size,
        full_calendar=feature_cfg.full_calendar,
        verbose=verbose,
        log_every_batches=10,
    )
    _log(verbose, f"[build] test user-day shape={test_features_df.shape}")

    # Persist
    _log(verbose, "[build] writing parquet artifacts...")
    train_features_df.to_parquet(train_features_path, index=False)
    test_features_df.to_parquet(test_features_path, index=False)
    labels_df.to_parquet(train_labels_path, index=False)
    _log(verbose, f"[build] wrote: {train_features_path.name}, {train_labels_path.name}, {test_features_path.name}")

    artifacts_path.write_text(
        json.dumps(
            {
                "vocab": vocab,
                "calendar_start_day": str(pd.to_datetime(calendar_start_day).date()),
                "calendar_num_days": int(calendar_num_days),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return BuildArtifacts(
        vocab=vocab,
        calendar_start_day=str(pd.to_datetime(calendar_start_day).date()),
        calendar_num_days=int(calendar_num_days),
        train_features_path=train_features_path,
        train_labels_path=train_labels_path,
        test_features_path=test_features_path,
    )


def materialize_splits(
    artifacts: BuildArtifacts,
    split_cfg: SplitConfig,
    output_dir: Path,
    overwrite: bool = True,
    *,
    verbose: bool = True,
) -> dict[str, Path]:
    _log(verbose, f"[split] output_dir={output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(verbose, "[split] loading cached features/labels...")
    train_features = pd.read_parquet(artifacts.train_features_path)
    train_labels = pd.read_parquet(artifacts.train_labels_path)
    test_features = pd.read_parquet(artifacts.test_features_path)
    _log(
        verbose,
        f"[split] loaded train_features={train_features.shape} train_labels={train_labels.shape} test_features={test_features.shape}",
    )

    train_user_ids = train_labels["userId"].tolist()
    train_users, val_users = split_users(train_user_ids, split_cfg=split_cfg)
    _log(verbose, f"[split] train_users={len(train_users):,} val_users={len(val_users):,}")

    val_user_set = set(val_users)
    is_val_row = train_features["userId"].isin(val_user_set)
    train_features_out = train_features[~is_val_row].reset_index(drop=True)
    val_features_out = train_features[is_val_row].reset_index(drop=True)

    is_val_label = train_labels["userId"].isin(val_user_set)
    train_labels_out = train_labels[~is_val_label].reset_index(drop=True)
    val_labels_out = train_labels[is_val_label].reset_index(drop=True)

    paths = {
        "train_features": output_dir / "train_user_day.parquet",
        "val_features": output_dir / "val_user_day.parquet",
        "test_features": output_dir / "test_user_day.parquet",
        "train_labels": output_dir / "train_labels.parquet",
        "val_labels": output_dir / "val_labels.parquet",
        "vocab": output_dir / "vocab.json",
        "calendar": output_dir / "calendar.json",
    }

    if overwrite:
        for p in paths.values():
            if p.exists():
                p.unlink()

    _log(verbose, "[split] writing split parquet files...")
    train_features_out.to_parquet(paths["train_features"], index=False)
    val_features_out.to_parquet(paths["val_features"], index=False)
    test_features.to_parquet(paths["test_features"], index=False)
    train_labels_out.to_parquet(paths["train_labels"], index=False)
    val_labels_out.to_parquet(paths["val_labels"], index=False)

    paths["vocab"].write_text(json.dumps(artifacts.vocab, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    paths["calendar"].write_text(
        json.dumps(
            {"start_day": artifacts.calendar_start_day, "num_days": int(artifacts.calendar_num_days)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return paths


def build_datasets_cli(
    train_path: str,
    test_path: str,
    output_dir: str | None,
    output_base_dir: str,
    cache_dir: str,
    cutoff_time: str | None,
    horizon_days: int,
    label_mode: str,
    drop_last_k_events: int,
    exclude_pages: str,
    exclude_auth_values: str,
    val_prob_low: float,
    val_prob_high: float,
    split_seed: str,
    batch_size: int,
    full_calendar: bool,
    force_recompute: bool,
    verbose: bool = True,
) -> dict[str, Path]:
    feature_cfg = FeatureBuildConfig(
        cutoff_time=_parse_cutoff_time(cutoff_time),
        horizon_days=horizon_days,
        label_mode="ever" if str(label_mode).lower() == "ever" else "horizon",
        drop_last_k_events=drop_last_k_events,
        exclude_pages=tuple([p.strip() for p in exclude_pages.split(",") if p.strip()]),
        exclude_auth_values=tuple([p.strip() for p in exclude_auth_values.split(",") if p.strip()]),
        full_calendar=full_calendar,
        batch_size=batch_size,
    )
    split_cfg = SplitConfig(val_prob_low=val_prob_low, val_prob_high=val_prob_high, split_seed=split_seed)

    if output_dir is None or str(output_dir).strip() == "":
        out_dir = suggest_output_dir(feature_cfg=feature_cfg, split_cfg=split_cfg, base_dir=Path(output_base_dir))
    else:
        out_dir = Path(output_dir)

    artifacts = build_and_cache_features(
        paths=DatasetPaths(train_path=Path(train_path), test_path=Path(test_path)),
        feature_cfg=feature_cfg,
        cache_dir=Path(cache_dir),
        force_recompute=force_recompute,
        verbose=verbose,
    )
    return materialize_splits(artifacts=artifacts, split_cfg=split_cfg, output_dir=out_dir, verbose=verbose)

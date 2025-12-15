from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowFeaturesConfig:
    """Configuration for turning a user-day table into fixed-length XGB features."""

    lookback_days: int = 51
    end_day: str | None = None  # YYYY-MM-DD; if None, infer from labels/user_day
    fill_value: float = 0.0
    dtype: str = "float32"


def _to_day(value: str | pd.Timestamp | np.datetime64) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]")
    return np.datetime64(pd.Timestamp(value).normalize().to_datetime64(), "D")


def infer_end_day(
    user_day: pd.DataFrame,
    labels: pd.DataFrame | None = None,
) -> np.datetime64:
    """
    Infer the inclusive window end day.

    Priority:
    1) If labels contain a single global cutoff_time, use cutoff_day - 1.
    2) Else, use max(user_day.day).
    """
    if labels is not None and "cutoff_time" in labels.columns:
        cutoff_days = pd.to_datetime(labels["cutoff_time"]).dt.normalize().dropna().unique()
        if len(cutoff_days) == 1:
            cutoff_day = _to_day(pd.Timestamp(cutoff_days[0]))
            end_day = cutoff_day - np.timedelta64(1, "D")
            if "day" in user_day.columns:
                max_day = _to_day(pd.to_datetime(user_day["day"]).max())
                if end_day <= max_day:
                    return end_day

    if "day" not in user_day.columns:
        raise ValueError("user_day DataFrame missing required column: day")
    return _to_day(pd.to_datetime(user_day["day"]).max())


def build_user_window_features(
    user_day_path: str | Path,
    *,
    labels_path: str | Path | None = None,
    cfg: WindowFeaturesConfig | None = None,
) -> pd.DataFrame:
    """
    Build fixed-length per-user features by flattening a user-day sequence.

    Returns a DataFrame indexed by userId with columns like:
      - d_events_t-00 (end_day)
      - d_events_t-01 (end_day - 1 day)
      - ...
    """
    cfg = cfg or WindowFeaturesConfig()
    lookback = int(cfg.lookback_days)
    if lookback <= 0:
        raise ValueError(f"lookback_days must be > 0, got {lookback}")

    user_day = pd.read_parquet(Path(user_day_path))
    user_day = user_day.sort_values(["userId", "day"], kind="mergesort")

    labels = None
    if labels_path is not None:
        labels = pd.read_parquet(Path(labels_path))

    end_day = _to_day(cfg.end_day) if cfg.end_day is not None else infer_end_day(user_day, labels=labels)
    start_day = end_day - np.timedelta64(lookback - 1, "D")

    day_d = pd.to_datetime(user_day["day"]).values.astype("datetime64[D]")
    in_window = (day_d >= start_day) & (day_d <= end_day)
    user_day = user_day.loc[in_window].copy()

    if user_day.empty:
        raise ValueError("No rows left after applying the window; check end_day/lookback_days.")

    lag = (end_day - day_d[in_window]).astype(int)
    user_day["lag"] = lag.astype(np.int16)

    feature_cols = [c for c in user_day.columns if c not in {"userId", "day", "lag"}]
    if not feature_cols:
        raise ValueError("No feature columns found (expected columns besides userId/day).")

    # Efficiency: features that are already window-aggregated or static (x_/u_) should
    # not be repeated across all lags. Keep them only at t-00.
    static_cols = [c for c in feature_cols if str(c).startswith(("x_", "u_"))]
    seq_cols = [c for c in feature_cols if c not in set(static_cols)]

    pad = len(str(lookback - 1))

    wide_parts = []
    if seq_cols:
        wide_seq = user_day.pivot(index="userId", columns="lag", values=seq_cols)
        full_cols = pd.MultiIndex.from_product([seq_cols, range(lookback)], names=["feature", "lag"])
        wide_seq = wide_seq.reindex(columns=full_cols)
        wide_seq.columns = [f"{feat}_t-{int(lag):0{pad}d}" for feat, lag in wide_seq.columns]
        wide_parts.append(wide_seq)

    if static_cols:
        t0 = user_day.loc[user_day["lag"] == 0, ["userId", *static_cols]].drop_duplicates("userId", keep="last")
        wide_static = t0.set_index("userId")[static_cols]
        wide_static.columns = [f"{c}_t-{0:0{pad}d}" for c in wide_static.columns]
        wide_parts.append(wide_static)

    wide = pd.concat(wide_parts, axis=1)
    wide = wide.fillna(float(cfg.fill_value))
    wide = wide.sort_index()
    wide = wide.reindex(sorted(wide.columns), axis=1)

    if cfg.dtype:
        wide = wide.astype(cfg.dtype, copy=False)
    return wide


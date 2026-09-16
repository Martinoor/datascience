"""Loading, validating, normalising and filtering raw event logs.

The raw Kaggle files (``train.parquet`` / ``test.parquet``) hold one row per
user event with 19 columns. The real training file has ~17.5M rows, so loading
is done in record batches with optional date-range and user-sampling filters
applied batch by batch, which keeps peak memory bounded.
"""

from __future__ import annotations

import datetime as dt
import io
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CHURN_PAGE = "Cancellation Confirmation"
# Pages/auth values that only appear at (or right before) cancellation. They
# must never be used as features, otherwise the label leaks into the inputs.
LEAKAGE_PAGES: tuple[str, ...] = ("Cancellation Confirmation", "Cancel")
LEAKAGE_AUTH: tuple[str, ...] = ("Cancelled",)

# Columns the app cannot work without. ``time`` may be derived from ``ts``.
REQUIRED_COLUMNS: tuple[str, ...] = ("userId", "sessionId", "page", "level")
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "time",
    "ts",
    "status",
    "gender",
    "auth",
    "itemInSession",
    "location",
    "userAgent",
    "method",
    "length",
    "song",
    "artist",
    "registration",
)
# Personal data that is never needed for analysis; dropped on load.
PII_COLUMNS: tuple[str, ...] = ("firstName", "lastName")
CATEGORICAL_COLUMNS: tuple[str, ...] = ("page", "level", "auth", "gender", "method", "device", "state")

SUPPORTED_FORMATS = ("parquet", "csv")
FileFormat = Literal["parquet", "csv"]
Source = str | Path | bytes | BinaryIO


class SchemaError(ValueError):
    """Raised when an input file does not look like a churn event log."""


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def infer_format(source: Source, fmt: str | None = None) -> FileFormat:
    """Return the file format, from ``fmt`` if given, else from the file name."""
    if fmt is not None:
        name = fmt
    elif isinstance(source, (str, Path)):
        name = Path(source).suffix
    else:
        name = Path(getattr(source, "name", "")).suffix
    name = name.lower().lstrip(".")
    if name == "pq":
        name = "parquet"
    if name not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported or unknown file format {name!r}; expected one of {SUPPORTED_FORMATS}")
    return name  # type: ignore[return-value]


def _as_readable(source: Source) -> str | BinaryIO:
    if isinstance(source, Path):
        return str(source)
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    return source


def _iter_batches(source: Source, fmt: FileFormat, batch_rows: int) -> Iterator[pd.DataFrame]:
    readable = _as_readable(source)
    if fmt == "parquet":
        pf = pq.ParquetFile(readable)
        available = set(pf.schema_arrow.names)
        wanted = [c for c in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if c in available]
        missing = [c for c in REQUIRED_COLUMNS if c not in available]
        if missing:
            raise SchemaError(f"Missing required columns: {missing}")
        for batch in pf.iter_batches(batch_size=batch_rows, columns=wanted):
            yield batch.to_pandas()
    else:
        # IDs must stay text: one blank userId would otherwise turn "1749042" into 1749042.0
        yield from pd.read_csv(readable, chunksize=batch_rows, low_memory=False, dtype={"userId": "string"})


def user_bucket(user_ids: pd.Series, buckets: int = 10_000) -> np.ndarray:
    """Deterministic bucket in ``[0, buckets)`` for each user id.

    ``pandas.util.hash_pandas_object`` uses a fixed hash key, so the same user
    always lands in the same bucket across runs, machines and batches.
    """
    hashed = pd.util.hash_pandas_object(user_ids.astype(str), index=False).to_numpy()
    return (hashed % np.uint64(buckets)).astype(np.int64)


def read_events(
    source: Source,
    *,
    fmt: str | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    user_fraction: float = 1.0,
    batch_rows: int = 1_000_000,
) -> pd.DataFrame:
    """Read an event log, applying cheap row filters batch by batch.

    Args:
        source: path, raw bytes or binary file-like object (e.g. a Streamlit upload).
        fmt: ``"parquet"`` or ``"csv"``; inferred from the file name when omitted.
        start, end: inclusive calendar-day bounds on the event time.
        user_fraction: keep a deterministic hash-based sample of users (0, 1].
        batch_rows: rows per record batch.
    """
    if not 0 < user_fraction <= 1:
        raise ValueError(f"user_fraction must be in (0, 1], got {user_fraction}")
    file_format = infer_format(source, fmt)

    parts: list[pd.DataFrame] = []
    for batch in _iter_batches(source, file_format, batch_rows):
        batch = batch.drop(columns=[c for c in PII_COLUMNS if c in batch.columns])
        if "time" not in batch.columns and "ts" in batch.columns:
            batch["time"] = pd.to_datetime(batch["ts"], unit="ms")
        if (start is not None or end is not None) and "time" in batch.columns:
            batch = batch[_date_mask(_to_datetime(batch["time"]), start, end)]
        if user_fraction < 1:
            keep = user_bucket(batch["userId"]) < round(user_fraction * 10_000)
            batch = batch[keep]
        parts.append(batch)

    if not parts:
        raise SchemaError("The file contains no rows")
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Validation and normalisation
# --------------------------------------------------------------------------- #
def validate_columns(columns: Iterable[str]) -> None:
    """Raise :class:`SchemaError` if required columns are missing."""
    cols = set(columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if "time" not in cols and "ts" not in cols:
        missing.append("time (or ts)")
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")


def _to_datetime(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.tz_localize(None) if values.dt.tz is not None else values
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="ms", errors="coerce")
    return pd.to_datetime(values, errors="coerce", format="mixed")


def detect_device(user_agent: pd.Series) -> pd.Series:
    """Coarse device family from the user-agent string.

    Vectorised equivalent of ``feature_pipeline.detect_device``; rules are
    checked in the same priority order (iPhone before Mac, etc.).
    """
    ua = user_agent.astype("string").str.lower()
    rules = [
        ("iPhone", "iphone"),
        ("iPad", "ipad"),
        ("Mac", "macintosh|mac os x"),
        ("Windows", "windows"),
        ("Android", "android"),
        ("Linux", "linux|ubuntu"),
    ]
    device = pd.Series("Other", index=user_agent.index, dtype="object")
    assigned = pd.Series(False, index=user_agent.index)
    for label, pattern in rules:
        hit = ua.str.contains(pattern, regex=True, na=False) & ~assigned
        device[hit] = label
        assigned |= hit
    return device


def parse_state(location: pd.Series) -> pd.Series:
    """Primary US state from ``"City-Metro, ST-ST2"`` style locations."""
    state = location.astype("string").str.rsplit(", ", n=1).str[-1].str.split("-").str[0]
    return state.fillna("Unknown").astype("object")


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, typed, sorted copy of a raw event log.

    - drops PII columns (first/last name);
    - derives ``time`` from epoch-millisecond ``ts`` when needed;
    - drops rows with no user id (logged-out traffic) or unparseable time;
    - derives ``device`` (from ``userAgent``) and ``state`` (from ``location``);
    - stores low-cardinality text columns as ``category`` to save memory;
    - sorts by ``userId`` then ``time``.
    """
    validate_columns(df.columns)
    out = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns]).copy()

    out["time"] = _to_datetime(out["time"] if "time" in out.columns else out["ts"])
    if pd.api.types.is_float_dtype(out["userId"]):  # integer ids with nulls, e.g. from other writers
        out["userId"] = out["userId"].astype("Int64")
    out["userId"] = out["userId"].astype("string").str.strip()
    out = out[out["userId"].notna() & (out["userId"] != "") & out["time"].notna()]

    if "registration" in out.columns:
        out["registration"] = _to_datetime(out["registration"])
    if "status" in out.columns:
        out["status"] = pd.to_numeric(out["status"], errors="coerce").astype("Int64")
    if "length" in out.columns:
        out["length"] = pd.to_numeric(out["length"], errors="coerce")
    out["sessionId"] = pd.to_numeric(out["sessionId"], errors="coerce").astype("Int64")

    if "userAgent" in out.columns:
        out["device"] = detect_device(out["userAgent"])
        out = out.drop(columns=["userAgent"])
    if "location" in out.columns:
        out["state"] = parse_state(out["location"])

    out["userId"] = out["userId"].astype(str)
    for col in CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("Unknown").astype("category")

    return out.sort_values(["userId", "time"], kind="mergesort").reset_index(drop=True)


def load_events(source: Source, **read_kwargs: object) -> pd.DataFrame:
    """Read (see :func:`read_events`) and normalise an event log in one call."""
    return normalize_events(read_events(source, **read_kwargs))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EventFilter:
    """Declarative filter over a normalised event log.

    Empty tuples mean "no constraint" for that dimension.
    """

    start: dt.date | None = None
    end: dt.date | None = None
    levels: tuple[str, ...] = ()
    genders: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    states: tuple[str, ...] = ()
    pages: tuple[str, ...] = ()
    user_ids: tuple[str, ...] = field(default=())
    exclude_leakage: bool = False

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"start ({self.start}) must not be after end ({self.end})")


def _date_mask(time: pd.Series, start: dt.date | None, end: dt.date | None) -> pd.Series:
    mask = pd.Series(True, index=time.index)
    if start is not None:
        mask &= time >= pd.Timestamp(start)
    if end is not None:
        mask &= time < pd.Timestamp(end) + pd.Timedelta(days=1)
    return mask


def filter_events(df: pd.DataFrame, flt: EventFilter) -> pd.DataFrame:
    """Apply an :class:`EventFilter`; returns a new DataFrame with a fresh index."""
    mask = _date_mask(df["time"], flt.start, flt.end)
    for column, allowed in (
        ("level", flt.levels),
        ("gender", flt.genders),
        ("device", flt.devices),
        ("state", flt.states),
        ("page", flt.pages),
        ("userId", flt.user_ids),
    ):
        if not allowed:
            continue
        if column not in df.columns:
            raise KeyError(f"Cannot filter on {column!r}: column not in data")
        mask &= df[column].astype(str).isin([str(v) for v in allowed])
    if flt.exclude_leakage:
        mask &= ~df["page"].astype(str).isin(LEAKAGE_PAGES)
        if "auth" in df.columns:
            mask &= ~df["auth"].astype(str).isin(LEAKAGE_AUTH)
    return df.loc[mask].reset_index(drop=True)


def filter_options(df: pd.DataFrame) -> dict[str, object]:
    """Values available for each filter widget (sorted), plus the date range."""
    options: dict[str, object] = {
        "min_date": df["time"].min().date() if len(df) else None,
        "max_date": df["time"].max().date() if len(df) else None,
    }
    for column in ("level", "gender", "device", "state", "page"):
        if column in df.columns:
            options[column] = sorted(str(v) for v in df[column].dropna().unique())
        else:
            options[column] = []
    return options


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def churn_labels(
    df: pd.DataFrame,
    cutoff: dt.date | pd.Timestamp | None = None,
    horizon_days: int = 10,
) -> pd.DataFrame:
    """Per-user churn labels.

    - ``cutoff=None`` ("ever" mode, as in the original project): a user churns
      if they ever reach the ``Cancellation Confirmation`` page.
    - with a cutoff ("horizon" mode, the competition definition): only users
      active before the cutoff and not yet churned are kept, and they churn if
      the confirmation happens in ``(cutoff, cutoff + horizon_days]``.

    Returns columns ``userId``, ``churn`` (0/1) and ``churn_time`` (NaT if none).
    """
    churn_time = df.loc[df["page"].astype(str) == CHURN_PAGE].groupby("userId", observed=True)["time"].min()
    first_seen = df.groupby("userId", observed=True)["time"].min()
    labels = pd.DataFrame({"userId": first_seen.index.astype(str)})
    labels["churn_time"] = labels["userId"].map(churn_time)

    if cutoff is None:
        labels["churn"] = labels["churn_time"].notna().astype("int8")
        return labels.reset_index(drop=True)

    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    cutoff_ts = pd.Timestamp(cutoff)
    horizon_end = cutoff_ts + pd.Timedelta(days=horizon_days)
    active_before = labels["userId"].map(first_seen) < cutoff_ts
    already_churned = labels["churn_time"] <= cutoff_ts
    labels = labels[active_before & ~already_churned.fillna(False)].copy()
    in_window = (labels["churn_time"] > cutoff_ts) & (labels["churn_time"] <= horizon_end)
    labels["churn"] = in_window.fillna(False).astype("int8")
    return labels.reset_index(drop=True)

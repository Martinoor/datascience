"""Tests for reading, validating and normalising event logs."""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pandas as pd
import pytest

from churn_app.data import (
    SchemaError,
    detect_device,
    infer_format,
    load_events,
    normalize_events,
    parse_state,
    read_events,
    user_bucket,
    validate_columns,
)


# --------------------------------------------------------------------------- #
# infer_format
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("source", "fmt", "expected"),
    [
        ("train.parquet", None, "parquet"),
        (Path("dir/TRAIN.PARQUET"), None, "parquet"),
        ("events.pq", None, "parquet"),
        ("events.csv", None, "csv"),
        (b"bytes", ".csv", "csv"),
        ("mislabelled.csv", "parquet", "parquet"),
    ],
)
def test_infer_format(source, fmt, expected):
    assert infer_format(source, fmt) == expected


def test_infer_format_uses_file_like_name():
    buf = io.BytesIO(b"")
    buf.name = "upload.parquet"
    assert infer_format(buf) == "parquet"


@pytest.mark.parametrize("source", ["notes.txt", "no_suffix", io.BytesIO(b"")])
def test_infer_format_rejects_unknown(source):
    with pytest.raises(ValueError, match="Unsupported"):
        infer_format(source)


# --------------------------------------------------------------------------- #
# read_events / load_events
# --------------------------------------------------------------------------- #
@pytest.fixture(params=["parquet", "csv"])
def saved_events(request, tmp_path: Path, raw_events: pd.DataFrame) -> Path:
    path = tmp_path / f"events.{request.param}"
    if request.param == "parquet":
        raw_events.to_parquet(path, index=False)
    else:
        raw_events.to_csv(path, index=False)
    return path


def test_read_events_from_path_drops_pii(saved_events: Path, raw_events: pd.DataFrame):
    df = read_events(saved_events)
    assert len(df) == len(raw_events)
    assert "firstName" not in df.columns
    assert "lastName" not in df.columns


def test_read_events_from_bytes_and_file_like(saved_events: Path):
    content = saved_events.read_bytes()
    fmt = saved_events.suffix
    from_bytes = read_events(content, fmt=fmt)
    buf = io.BytesIO(content)
    buf.name = saved_events.name
    from_buffer = read_events(buf)
    assert len(from_bytes) == len(from_buffer) == 13


def test_load_events_round_trip_matches_in_memory_normalisation(saved_events: Path, events: pd.DataFrame):
    loaded = load_events(saved_events)
    cols = ["userId", "page", "sessionId", "level", "device", "state"]
    pd.testing.assert_frame_equal(
        loaded[cols].astype(str).reset_index(drop=True), events[cols].astype(str).reset_index(drop=True)
    )
    assert (loaded["time"].to_numpy() == events["time"].to_numpy()).all()


def test_read_events_derives_time_from_ts(tmp_path: Path, raw_events: pd.DataFrame):
    path = tmp_path / "ts_only.parquet"
    raw_events.drop(columns=["time"]).to_parquet(path)
    df = read_events(path)
    assert (df["time"].to_numpy() == raw_events["time"].to_numpy()).all()


def test_read_events_date_range_is_inclusive_by_day(saved_events: Path):
    df = read_events(saved_events, start=dt.date(2018, 10, 2), end=dt.date(2018, 10, 10))
    times = pd.to_datetime(df["time"])
    assert times.min() == pd.Timestamp("2018-10-02 20:00:00")
    # the whole of the end day is included
    assert times.max() == pd.Timestamp("2018-10-10 08:01:10")
    assert len(df) == 8


def test_read_events_missing_required_parquet_columns(tmp_path: Path, raw_events: pd.DataFrame):
    path = tmp_path / "bad.parquet"
    raw_events.drop(columns=["page"]).to_parquet(path)
    with pytest.raises(SchemaError, match="page"):
        read_events(path)


def test_load_events_missing_time_and_ts(tmp_path: Path, raw_events: pd.DataFrame):
    path = tmp_path / "no_time.csv"
    raw_events.drop(columns=["time", "ts"]).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="time"):
        load_events(path)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.5])
def test_read_events_rejects_bad_fraction(saved_events: Path, fraction: float):
    with pytest.raises(ValueError, match="user_fraction"):
        read_events(saved_events, user_fraction=fraction)


def test_user_sampling_is_deterministic_and_keeps_whole_users(tmp_path: Path, sample_raw: pd.DataFrame):
    path = tmp_path / "sample.parquet"
    sample_raw.to_parquet(path)
    a = read_events(path, user_fraction=0.3, batch_rows=10_000)  # many batches
    b = read_events(path, user_fraction=0.3, batch_rows=10**9)  # one batch
    assert set(a["userId"]) == set(b["userId"])
    kept = set(a["userId"])
    assert 0.15 < len(kept) / sample_raw["userId"].nunique() < 0.45
    # every event of a sampled user is kept
    assert len(a) == sample_raw["userId"].isin(kept).sum()


def test_user_bucket_is_stable():
    ids = pd.Series(["1749042", "1465194", "1749042"])
    buckets = user_bucket(ids)
    assert buckets[0] == buckets[2]
    assert ((buckets >= 0) & (buckets < 10_000)).all()
    assert (user_bucket(ids) == buckets).all()


# --------------------------------------------------------------------------- #
# validation / normalisation
# --------------------------------------------------------------------------- #
def test_validate_columns_lists_everything_missing():
    with pytest.raises(SchemaError) as exc:
        validate_columns(["userId", "page"])
    message = str(exc.value)
    assert "sessionId" in message
    assert "level" in message
    assert "time (or ts)" in message


def test_validate_columns_accepts_ts_instead_of_time():
    validate_columns(["userId", "sessionId", "page", "level", "ts"])


def test_normalize_drops_anonymous_and_unparseable_rows(raw_events: pd.DataFrame):
    raw = raw_events.copy()
    raw["time"] = raw["time"].astype(str)
    raw.loc[0, "time"] = "not a date"
    out = normalize_events(raw)
    assert "" not in set(out["userId"])
    assert len(out) == len(raw) - 2


def test_normalize_types_sorting_and_derived_columns(events: pd.DataFrame):
    assert pd.api.types.is_datetime64_any_dtype(events["time"])
    assert pd.api.types.is_datetime64_any_dtype(events["registration"])
    for col in ("page", "level", "auth", "gender", "device", "state"):
        assert isinstance(events[col].dtype, pd.CategoricalDtype), col
    assert "userAgent" not in events.columns
    assert list(events["userId"].unique()) == ["1", "2", "3"]
    assert events.groupby("userId")["time"].apply(lambda t: t.is_monotonic_increasing).all()
    by_user = events.groupby("userId", observed=True)[["device", "state"]].first().astype(str)
    assert by_user.loc["1"].tolist() == ["Windows", "TX"]
    assert by_user.loc["2"].tolist() == ["iPhone", "NY"]
    assert by_user.loc["3"].tolist() == ["Mac", "WA"]


def test_normalize_handles_timezone_aware_and_epoch_times(raw_events: pd.DataFrame):
    aware = raw_events.assign(time=raw_events["time"].dt.tz_localize("UTC"))
    epoch = raw_events.drop(columns=["time"])
    a, b = normalize_events(aware), normalize_events(epoch)
    assert a["time"].dt.tz is None
    assert (a["time"].to_numpy() == b["time"].to_numpy()).all()


def test_normalize_turns_float_user_ids_back_into_integer_strings(raw_events: pd.DataFrame):
    raw = raw_events.assign(userId=pd.to_numeric(raw_events["userId"].replace("", None)))
    assert pd.api.types.is_float_dtype(raw["userId"])
    assert list(normalize_events(raw)["userId"].unique()) == ["1", "2", "3"]


def test_normalize_does_not_mutate_input(raw_events: pd.DataFrame):
    before = raw_events.copy()
    normalize_events(raw_events)
    pd.testing.assert_frame_equal(raw_events, before)


@pytest.mark.parametrize(
    ("agent", "expected"),
    [
        ("Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X)", "iPhone"),
        ("Mozilla/5.0 (iPad; CPU OS 7_1_2 like Mac OS X)", "iPad"),
        ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4)", "Mac"),
        ("Mozilla/5.0 (Windows NT 6.1; WOW64)", "Windows"),
        ("Mozilla/5.0 (Linux; Android 4.4)", "Android"),
        ("Mozilla/5.0 (X11; Ubuntu; Linux x86_64)", "Linux"),
        ("curl/7.0", "Other"),
        (None, "Other"),
    ],
)
def test_detect_device(agent, expected):
    assert detect_device(pd.Series([agent], dtype="object")).iloc[0] == expected


def test_parse_state():
    locations = pd.Series(["Dallas-Fort Worth-Arlington, TX", "New York-Newark-Jersey City, NY-NJ-PA", None])
    assert parse_state(locations).tolist() == ["TX", "NY", "Unknown"]

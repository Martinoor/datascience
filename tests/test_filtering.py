"""Tests for EventFilter / filter_events / filter_options / churn_labels."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from churn_app.data import EventFilter, churn_labels, filter_events, filter_options


def test_empty_filter_keeps_everything(events: pd.DataFrame):
    out = filter_events(events, EventFilter())
    assert len(out) == len(events)
    assert out.index.equals(pd.RangeIndex(len(events)))


def test_filter_does_not_mutate_input(events: pd.DataFrame):
    before = events.copy()
    filter_events(events, EventFilter(levels=("paid",), exclude_leakage=True))
    pd.testing.assert_frame_equal(events, before)


def test_start_after_end_is_rejected():
    with pytest.raises(ValueError, match="must not be after"):
        EventFilter(start=dt.date(2018, 11, 1), end=dt.date(2018, 10, 1))


@pytest.mark.parametrize(
    ("start", "end", "expected_rows"),
    [
        (dt.date(2018, 10, 1), dt.date(2018, 10, 1), 3),  # a single day, fully included
        (dt.date(2018, 10, 5), None, 7),
        (None, dt.date(2018, 10, 4), 5),
        (dt.date(2018, 11, 15), dt.date(2018, 11, 15), 1),  # 23:59:59 still on the end day
        (dt.date(2018, 12, 1), None, 0),
    ],
)
def test_date_filter_is_inclusive_by_calendar_day(events, start, end, expected_rows):
    assert len(filter_events(events, EventFilter(start=start, end=end))) == expected_rows


@pytest.mark.parametrize(
    ("flt", "expected_users"),
    [
        (EventFilter(levels=("free",)), {"2"}),
        (EventFilter(genders=("M",)), {"1", "3"}),
        (EventFilter(devices=("iPhone", "Mac")), {"2", "3"}),
        (EventFilter(states=("TX",)), {"1"}),
        (EventFilter(pages=("Help",)), {"3"}),
        (EventFilter(user_ids=("1", "3")), {"1", "3"}),
        (EventFilter(levels=("paid",), states=("WA",)), {"3"}),  # dimensions combine with AND
        (EventFilter(levels=("free",), states=("TX",)), set()),
    ],
)
def test_categorical_filters(events, flt, expected_users):
    assert set(filter_events(events, flt)["userId"]) == expected_users


def test_filter_values_within_a_dimension_are_ored(events):
    out = filter_events(events, EventFilter(pages=("NextSong", "Help")))
    assert set(out["page"].astype(str)) == {"NextSong", "Help"}
    assert len(out) == 5


def test_exclude_leakage_drops_cancellation_rows(events):
    out = filter_events(events, EventFilter(exclude_leakage=True))
    assert not out["page"].astype(str).isin(["Cancel", "Cancellation Confirmation"]).any()
    assert not (out["auth"].astype(str) == "Cancelled").any()
    assert len(out) == len(events) - 2


def test_filter_on_missing_column_raises(events):
    with pytest.raises(KeyError, match="gender"):
        filter_events(events.drop(columns=["gender"]), EventFilter(genders=("F",)))


def test_filter_options(events):
    opts = filter_options(events)
    assert opts["min_date"] == dt.date(2018, 10, 1)
    assert opts["max_date"] == dt.date(2018, 11, 19)
    assert opts["level"] == ["free", "paid"]
    assert opts["device"] == ["Mac", "Windows", "iPhone"]
    assert "Cancellation Confirmation" in opts["page"]


def test_filter_options_on_empty_and_partial_frames(events):
    opts = filter_options(events.iloc[0:0].drop(columns=["state"]))
    assert opts["min_date"] is None
    assert opts["state"] == []


# --------------------------------------------------------------------------- #
# churn_labels
# --------------------------------------------------------------------------- #
def _as_dict(labels: pd.DataFrame) -> dict[str, int]:
    return dict(zip(labels["userId"], labels["churn"], strict=True))


def test_labels_ever_mode(events):
    labels = churn_labels(events)
    assert _as_dict(labels) == {"1": 0, "2": 1, "3": 0}
    assert labels.loc[labels["userId"] == "2", "churn_time"].iloc[0] == pd.Timestamp("2018-10-10 08:01:10")
    assert labels.loc[labels["userId"] == "1", "churn_time"].isna().all()


def test_labels_horizon_mode_positive_inside_window(events):
    # user 3 is first seen after the cutoff and is therefore excluded
    labels = churn_labels(events, cutoff=dt.date(2018, 10, 8), horizon_days=10)
    assert _as_dict(labels) == {"1": 0, "2": 1}


def test_labels_horizon_mode_negative_outside_window(events):
    labels = churn_labels(events, cutoff=dt.date(2018, 10, 3), horizon_days=5)  # window ends 10-08
    assert _as_dict(labels) == {"1": 0, "2": 0}


def test_labels_horizon_mode_drops_users_already_churned(events):
    labels = churn_labels(events, cutoff=pd.Timestamp("2018-10-11"), horizon_days=10)
    assert _as_dict(labels) == {"1": 0}


def test_labels_horizon_window_end_is_inclusive(events):
    churn_at = pd.Timestamp("2018-10-10 08:01:10")
    labels = churn_labels(events, cutoff=churn_at - pd.Timedelta(days=2), horizon_days=2)
    assert _as_dict(labels)["2"] == 1


@pytest.mark.parametrize("horizon", [0, -3])
def test_labels_reject_non_positive_horizon(events, horizon):
    with pytest.raises(ValueError, match="horizon_days"):
        churn_labels(events, cutoff=dt.date(2018, 10, 8), horizon_days=horizon)


def test_labels_on_sample_data_match_cancellation_events(sample_events):
    labels = churn_labels(sample_events)
    cancelled = set(sample_events.loc[sample_events["page"] == "Cancellation Confirmation", "userId"])
    assert set(labels.loc[labels["churn"] == 1, "userId"]) == cancelled
    assert labels["userId"].is_unique

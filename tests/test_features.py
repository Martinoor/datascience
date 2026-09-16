from __future__ import annotations

import pandas as pd
import pytest

from churn_app.data import churn_labels
from churn_app.features import (
    build_user_features,
    churn_rate_by,
    daily_activity,
    page_distribution,
    user_timeline,
)


def test_user_features_counts(events):
    f = build_user_features(events)
    assert list(f.index) == ["1", "2", "3"]
    u1 = f.loc["1"]
    assert u1["n_events"] == 5
    assert u1["n_sessions"] == 2
    assert u1["active_days"] == 2
    assert u1["songs_played"] == 2
    assert u1["thumbs_up"] == 1
    assert u1["thumbs_down_ratio"] == pytest.approx(0.5)
    assert u1["is_paid_now"] == 1
    assert u1["is_male"] == 1
    assert u1["listening_hours"] == pytest.approx(360 / 3600)
    assert f.loc["2", "http_404_rate"] == pytest.approx(1 / 3)
    assert (f.dtypes == "float64").all()


def test_user_features_never_use_cancellation_events(events):
    f = build_user_features(events)
    # user 2 has 5 events, 2 of which are Cancel / Cancellation Confirmation
    assert f.loc["2", "n_events"] == 3
    assert f.loc["2", "adverts"] == 1
    assert f.loc["2", "errors"] == 1


def test_user_features_respect_reference_time(events):
    f = build_user_features(events, reference=pd.Timestamp("2018-10-05"))
    assert set(f.index) == {"1", "2"}  # user 3 has no events yet
    assert f.loc["1", "n_events"] == 3
    assert f.loc["1", "days_since_last_event"] == pytest.approx(
        (pd.Timestamp("2018-10-05") - pd.Timestamp("2018-10-01 10:04")).total_seconds() / 86_400
    )
    assert f.loc["1", "tenure_days"] == pytest.approx(34.0)


def test_user_features_empty_when_no_events_before_reference(events):
    f = build_user_features(events, reference=pd.Timestamp("2018-01-01"))
    assert f.empty
    assert f.index.name == "userId"


def test_features_are_identical_with_or_without_future_events(sample_events):
    cutoff = pd.Timestamp("2018-11-01")
    full = build_user_features(sample_events, cutoff)
    past_only = build_user_features(sample_events[sample_events["time"] < cutoff], cutoff)
    pd.testing.assert_frame_equal(full, past_only)


def test_daily_activity(events):
    daily = daily_activity(events)
    assert daily["events"].sum() == len(events)
    row = daily.set_index("day").loc[pd.Timestamp("2018-10-10")]
    assert row["events"] == 3
    assert row["active_users"] == 1
    assert daily_activity(events.iloc[0:0]).empty


def test_page_distribution(events):
    pages = page_distribution(events)
    assert pages.iloc[0].tolist()[:2] == ["NextSong", 4]
    assert pages["share"].sum() == pytest.approx(1.0)


def test_churn_rate_by_binary_and_quantiles(sample_events):
    features = build_user_features(sample_events)
    labels = churn_labels(sample_events)
    binary = churn_rate_by(features, labels, "is_paid_now")
    assert set(binary["bucket"]) == {"0", "1"}
    assert binary["users"].sum() == len(features)
    buckets = churn_rate_by(features, labels, "n_events", bins=4)
    assert len(buckets) == 4
    assert buckets["churn_rate"].between(0, 1).all()


def test_churn_rate_by_without_overlap(events):
    labels = pd.DataFrame({"userId": ["x"], "churn": [1]})
    assert churn_rate_by(build_user_features(events), labels, "n_events").empty


def test_user_timeline(events):
    tl = user_timeline(events, "1")
    assert tl["events"].sum() == 5
    assert set(tl["page"]) == {"NextSong", "Thumbs Up", "Home", "Thumbs Down"}
    assert user_timeline(events, "nope").empty

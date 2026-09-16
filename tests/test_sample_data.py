from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn_app.data import load_events
from churn_app.sample_data import generate_events, main

KAGGLE_COLUMNS = {
    "status", "gender", "firstName", "level", "lastName", "userId", "ts", "auth", "page", "sessionId",
    "location", "itemInSession", "userAgent", "method", "length", "song", "artist", "time", "registration",
}  # fmt: skip


def test_schema_matches_kaggle(sample_raw):
    assert set(sample_raw.columns) == KAGGLE_COLUMNS
    assert str(sample_raw["time"].dtype) == "datetime64[us]"
    assert sample_raw["time"].min() >= pd.Timestamp("2018-10-01")
    assert sample_raw["time"].max() < pd.Timestamp("2018-11-20")


def test_generation_is_deterministic():
    pd.testing.assert_frame_equal(generate_events(40, seed=1), generate_events(40, seed=1))
    assert not generate_events(40, seed=1).equals(generate_events(40, seed=2))


def test_cancellation_is_each_churners_last_event(sample_raw):
    ordered = sample_raw.sort_values(["userId", "ts"], kind="mergesort")
    last = ordered.groupby("userId").tail(1)
    churners = set(sample_raw.loc[sample_raw["page"] == "Cancellation Confirmation", "userId"])
    assert churners, "sample should contain churners"
    assert set(last.loc[last["page"] == "Cancellation Confirmation", "userId"]) == churners
    assert (sample_raw.loc[sample_raw["page"] == "Cancellation Confirmation", "auth"] == "Cancelled").all()


def test_rejects_non_positive_users():
    with pytest.raises(ValueError):
        generate_events(0)


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_cli_writes_loadable_file(tmp_path: Path, capsys, suffix):
    out = tmp_path / f"nested/sample{suffix}"
    main(["--users", "30", "--seed", "3", "--out", str(out)])
    assert "Wrote" in capsys.readouterr().out
    assert load_events(out)["userId"].nunique() == 30

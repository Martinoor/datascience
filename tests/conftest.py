from __future__ import annotations

import pandas as pd
import pytest

from churn_app.data import normalize_events
from churn_app.sample_data import generate_events

WINDOWS_UA = '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0 Safari/537.36"'
IPHONE_UA = '"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 Mobile/11D257"'
MAC_UA = '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 Safari/537.77.4"'

# (userId, time, page, sessionId, level, gender, auth, status, location, userAgent, length)
_ROWS = [
    # user 1: paid, never cancels, two sessions
    ("1", "2018-10-01 10:00:00", "NextSong", 11, "paid", "M", "Logged In", 200, "Dallas-Fort Worth-Arlington, TX", WINDOWS_UA, 200.0),
    ("1", "2018-10-01 10:03:20", "Thumbs Up", 11, "paid", "M", "Logged In", 307, "Dallas-Fort Worth-Arlington, TX", WINDOWS_UA, None),
    ("1", "2018-10-01 10:04:00", "Home", 11, "paid", "M", "Logged In", 200, "Dallas-Fort Worth-Arlington, TX", WINDOWS_UA, None),
    ("1", "2018-10-05 09:00:00", "Thumbs Down", 12, "paid", "M", "Logged In", 307, "Dallas-Fort Worth-Arlington, TX", WINDOWS_UA, None),
    ("1", "2018-10-05 09:01:00", "NextSong", 12, "paid", "M", "Logged In", 200, "Dallas-Fort Worth-Arlington, TX", WINDOWS_UA, 160.0),
    # user 2: free, cancels on 2018-10-10
    ("2", "2018-10-02 20:00:00", "NextSong", 21, "free", "F", "Logged In", 200, "New York-Newark-Jersey City, NY-NJ-PA", IPHONE_UA, 180.0),
    ("2", "2018-10-02 20:03:00", "Roll Advert", 21, "free", "F", "Logged In", 200, "New York-Newark-Jersey City, NY-NJ-PA", IPHONE_UA, None),
    ("2", "2018-10-10 08:00:00", "Error", 22, "free", "F", "Logged In", 404, "New York-Newark-Jersey City, NY-NJ-PA", IPHONE_UA, None),
    ("2", "2018-10-10 08:01:00", "Cancel", 22, "free", "F", "Logged In", 307, "New York-Newark-Jersey City, NY-NJ-PA", IPHONE_UA, None),
    ("2", "2018-10-10 08:01:10", "Cancellation Confirmation", 22, "free", "F", "Cancelled", 200, "New York-Newark-Jersey City, NY-NJ-PA", IPHONE_UA, None),
    # logged-out traffic without a user id: must be dropped
    ("", "2018-10-03 12:00:00", "Home", 99, "free", None, "Logged Out", 200, None, None, None),
    # user 3: first seen late (after typical cutoffs)
    ("3", "2018-11-15 23:59:59", "NextSong", 31, "paid", "M", "Logged In", 200, "Seattle-Tacoma-Bellevue, WA", MAC_UA, 240.0),
    ("3", "2018-11-19 07:00:00", "Help", 32, "paid", "M", "Logged In", 200, "Seattle-Tacoma-Bellevue, WA", MAC_UA, None),
]  # fmt: skip
_COLUMNS = [
    "userId",
    "time",
    "page",
    "sessionId",
    "level",
    "gender",
    "auth",
    "status",
    "location",
    "userAgent",
    "length",
]


@pytest.fixture
def raw_events() -> pd.DataFrame:
    """Small hand-written raw log in the Kaggle schema (incl. PII + ts columns)."""
    df = pd.DataFrame(_ROWS, columns=_COLUMNS)
    df["time"] = pd.to_datetime(df["time"])
    df["ts"] = df["time"].astype("datetime64[ms]").astype("int64")
    df["registration"] = pd.Timestamp("2018-09-01")
    df["firstName"] = "Ada"
    df["lastName"] = "Lovelace"
    return df


@pytest.fixture
def events(raw_events: pd.DataFrame) -> pd.DataFrame:
    return normalize_events(raw_events)


@pytest.fixture(scope="session")
def sample_raw() -> pd.DataFrame:
    return generate_events(n_users=250, seed=11)


@pytest.fixture(scope="session")
def sample_events(sample_raw: pd.DataFrame) -> pd.DataFrame:
    return normalize_events(sample_raw)

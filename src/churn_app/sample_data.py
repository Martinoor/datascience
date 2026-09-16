"""Deterministic synthetic event logs with the same schema as the Kaggle data.

The competition data can't be redistributed, so the app, the tests and CI all
use this generator. Output is fully determined by the arguments, so the same
seed always yields the same rows.

Usage::

    python -m churn_app.sample_data --users 500 --out data/sample_events.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2018-10-01")
END = pd.Timestamp("2018-11-20")

PAGES = np.array(
    [
        "NextSong",
        "Thumbs Up",
        "Home",
        "Add to Playlist",
        "Roll Advert",
        "Add Friend",
        "Logout",
        "Thumbs Down",
        "Downgrade",
        "Settings",
        "Help",
        "Upgrade",
        "About",
        "Save Settings",
        "Error",
        "Submit Downgrade",
        "Submit Upgrade",
    ]
)
# Approximate page mix of the real training data (NextSong ~82%).
BASE_PAGE_P = np.array(
    [
        0.80,
        0.045,
        0.037,
        0.023,
        0.016,
        0.015,
        0.012,
        0.009,
        0.007,
        0.006,
        0.005,
        0.0022,
        0.002,
        0.0012,
        0.001,
        0.0003,
        0.0006,
    ]
)
SUBMIT_PAGES = ("Submit Upgrade", "Submit Downgrade")
COLUMNS = (
    "status", "gender", "firstName", "level", "lastName", "userId", "ts", "auth", "page", "sessionId",
    "location", "itemInSession", "userAgent", "method", "length", "song", "artist", "registration",
)  # fmt: skip
GET_PAGES = {"Home", "Help", "About", "Settings", "Downgrade", "Upgrade", "Error", "Roll Advert"}

LOCATIONS = [
    "Dallas-Fort Worth-Arlington, TX",
    "New York-Newark-Jersey City, NY-NJ-PA",
    "Los Angeles-Long Beach-Anaheim, CA",
    "Chicago-Naperville-Elgin, IL-IN-WI",
    "Miami-Fort Lauderdale-West Palm Beach, FL",
    "Seattle-Tacoma-Bellevue, WA",
    "Boston-Cambridge-Newton, MA-NH",
    "Phoenix-Mesa-Scottsdale, AZ",
]
USER_AGENTS = [
    '"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36"',
    '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.77.4 (KHTML, like Gecko) Version/7.0.5 Safari/537.77.4"',
    '"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"',
    '"Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0"',
    '"Mozilla/5.0 (iPad; CPU OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53"',
]


def _page_probs(churner: bool) -> np.ndarray:
    p = BASE_PAGE_P.copy()
    if churner:  # mild, realistic signal: more ads/thumbs-down/help, fewer thumbs-up
        p[PAGES == "Thumbs Down"] *= 1.8
        p[PAGES == "Roll Advert"] *= 1.6
        p[PAGES == "Help"] *= 1.4
        p[PAGES == "Downgrade"] *= 1.5
        p[PAGES == "Thumbs Up"] *= 0.8
    return p / p.sum()


def generate_events(
    n_users: int = 300,
    churn_rate: float = 0.22,
    seed: int = 7,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
) -> pd.DataFrame:
    """Generate a raw (un-normalised) event log mimicking ``train.parquet``."""
    if n_users <= 0:
        raise ValueError("n_users must be positive")
    rng = np.random.default_rng(seed)
    span_s = (end - start).total_seconds()
    cols: dict[str, list[np.ndarray]] = {c: [] for c in COLUMNS}
    session_id = 1

    for u in range(n_users):
        user_id = str(1_000_000 + u * 37)
        churner = rng.random() < churn_rate
        gender = rng.choice(["F", "M"])
        level = "paid" if rng.random() < 0.6 else "free"
        location = LOCATIONS[rng.integers(len(LOCATIONS))]
        agent = USER_AGENTS[rng.integers(len(USER_AGENTS))]
        registration = (start - pd.Timedelta(days=float(rng.uniform(1, 300)))).value // 1_000_000
        first_offset = rng.uniform(0, 0.3) * span_s
        last_offset = rng.uniform(0.3, 1.0) * span_s if churner else span_s
        sessions_per_day = rng.lognormal(mean=-0.7, sigma=0.6)
        days = max((last_offset - first_offset) / 86_400, 0.5)
        n_sessions = max(1, rng.poisson(sessions_per_day * days))
        starts = np.sort(rng.uniform(first_offset, last_offset, size=n_sessions))
        probs = _page_probs(churner)
        cursor = -np.inf  # end of the previous session: sessions never overlap

        for s_idx, s_start in enumerate(starts):
            s_start = max(s_start, cursor + rng.exponential(3_600))
            n_ev = int(rng.geometric(1 / 25))
            pages = rng.choice(PAGES, size=n_ev, p=probs).astype(object)
            gaps = np.where(pages == "NextSong", rng.normal(240, 50, n_ev).clip(30), rng.exponential(20, n_ev))
            offsets = s_start + np.concatenate([[0.0], np.cumsum(gaps[:-1])])
            is_last = churner and s_idx == len(starts) - 1
            if is_last:
                pages = np.append(pages, ["Cancel", "Cancellation Confirmation"])
                offsets = np.append(offsets, [offsets[-1] + 30, offsets[-1] + 40])
            n = len(pages)
            cursor = offsets[-1]
            keep = offsets < span_s
            if not keep.any():
                continue
            song_ids = rng.integers(0, 400, n)
            is_song = pages == "NextSong"
            auth = np.full(n, "Logged In", dtype=object)
            if is_last:
                auth[-1] = "Cancelled"
            levels = np.full(n, level, dtype=object)
            changed = np.flatnonzero(np.isin(pages, ["Submit Upgrade", "Submit Downgrade"]))
            for i in changed:  # a submitted plan change takes effect from the next event
                level = "paid" if pages[i] == "Submit Upgrade" else "free"
                levels[i + 1 :] = level

            values = {
                "status": np.where(
                    pages == "Error", 404, np.where(np.isin(pages, ["Logout", "Cancel", *SUBMIT_PAGES]), 307, 200)
                ),
                "gender": np.full(n, gender, dtype=object),
                "firstName": np.full(n, "Synthetic", dtype=object),
                "level": levels,
                "lastName": np.full(n, f"User{u}", dtype=object),
                "userId": np.full(n, user_id, dtype=object),
                "ts": start.value // 1_000_000 + (offsets * 1000).astype(np.int64),
                "auth": auth,
                "page": pages,
                "sessionId": np.full(n, session_id, dtype=np.int64),
                "location": np.full(n, location, dtype=object),
                "itemInSession": np.arange(n, dtype=np.int64),
                "userAgent": np.full(n, agent, dtype=object),
                "method": np.where(np.isin(pages, list(GET_PAGES)), "GET", "PUT").astype(object),
                "length": np.where(is_song, rng.normal(245, 60, n).clip(30).round(5), np.nan),
                "song": np.where(is_song, np.char.add("Song ", song_ids.astype(str)), None),
                "artist": np.where(is_song, np.char.add("Artist ", (song_ids // 8).astype(str)), None),
                "registration": np.full(n, registration, dtype=np.int64),
            }
            for name, arr in values.items():
                cols[name].append(arr[keep])
            session_id += 1

    df = pd.DataFrame({name: np.concatenate(parts) for name, parts in cols.items()})
    df["time"] = pd.to_datetime(df["ts"], unit="ms").astype("datetime64[us]")
    df["registration"] = pd.to_datetime(df["registration"], unit="ms").astype("datetime64[us]")
    return df.sort_values(["ts", "userId"], kind="mergesort").reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--churn-rate", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("data/sample_events.parquet"))
    args = parser.parse_args(argv)

    df = generate_events(args.users, args.churn_rate, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".csv":
        df.to_csv(args.out, index=False)
    else:
        df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df):,} events for {df['userId'].nunique()} users to {args.out}")


if __name__ == "__main__":
    main()

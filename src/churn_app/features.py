"""User-level feature engineering and aggregations used by the app.

Features are computed from events strictly *before* a reference time, and
cancellation pages are always excluded, so labels can't leak into inputs.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from churn_app.data import LEAKAGE_AUTH, LEAKAGE_PAGES

# Page -> feature name for per-user page counts.
PAGE_COUNT_FEATURES: dict[str, str] = {
    "NextSong": "songs_played",
    "Thumbs Up": "thumbs_up",
    "Thumbs Down": "thumbs_down",
    "Add to Playlist": "add_to_playlist",
    "Add Friend": "add_friend",
    "Roll Advert": "adverts",
    "Help": "help",
    "Settings": "settings",
    "Error": "errors",
    "Upgrade": "upgrade_views",
    "Downgrade": "downgrade_views",
    "Submit Upgrade": "submit_upgrade",
    "Submit Downgrade": "submit_downgrade",
    "Logout": "logouts",
}
RECENT_DAYS = 7


def _feature_events(events: pd.DataFrame, reference: pd.Timestamp) -> pd.DataFrame:
    mask = (events["time"] < reference) & ~events["page"].astype(str).isin(LEAKAGE_PAGES)
    if "auth" in events.columns:
        mask &= ~events["auth"].astype(str).isin(LEAKAGE_AUTH)
    return events.loc[mask]


def build_user_features(
    events: pd.DataFrame,
    reference: dt.date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per user (indexed by ``userId``) of numeric behavioural features.

    Args:
        events: normalised events (see :func:`churn_app.data.normalize_events`).
        reference: features use events strictly before this time. Defaults to
            one second after the last event, i.e. use everything.
    """
    reference_ts = events["time"].max() + pd.Timedelta(seconds=1) if reference is None else pd.Timestamp(reference)

    ev = _feature_events(events, reference_ts)
    if ev.empty:
        return pd.DataFrame(index=pd.Index([], name="userId", dtype=str))

    g = ev.groupby("userId", observed=True)
    feats = pd.DataFrame(
        {
            "n_events": g.size(),
            "n_sessions": g["sessionId"].nunique(),
            "active_days": ev["time"].dt.normalize().groupby(ev["userId"], observed=True).nunique(),
            "first_event": g["time"].min(),
            "last_event": g["time"].max(),
        }
    )
    feats["days_since_last_event"] = (reference_ts - feats["last_event"]).dt.total_seconds() / 86_400
    feats["days_observed"] = (feats["last_event"] - feats["first_event"]).dt.total_seconds() / 86_400
    feats["events_per_session"] = feats["n_events"] / feats["n_sessions"].clip(lower=1)
    feats["events_per_active_day"] = feats["n_events"] / feats["active_days"].clip(lower=1)

    page_counts = pd.crosstab(ev["userId"].astype(str), ev["page"].astype(str))
    for page, name in PAGE_COUNT_FEATURES.items():
        feats[name] = page_counts[page] if page in page_counts.columns else 0
    feats[list(PAGE_COUNT_FEATURES.values())] = feats[list(PAGE_COUNT_FEATURES.values())].fillna(0)

    n = feats["n_events"].clip(lower=1)
    feats["thumbs_down_ratio"] = feats["thumbs_down"] / (feats["thumbs_up"] + feats["thumbs_down"]).clip(lower=1)
    feats["advert_rate"] = feats["adverts"] / n
    feats["error_rate"] = feats["errors"] / n
    feats["help_rate"] = feats["help"] / n
    feats["downgrade_rate"] = feats["downgrade_views"] / n

    recent = ev[ev["time"] >= reference_ts - pd.Timedelta(days=RECENT_DAYS)]
    feats["events_last_7d"] = recent.groupby("userId", observed=True).size().reindex(feats.index).fillna(0)
    weeks_observed = (feats["days_observed"] / RECENT_DAYS).clip(lower=1)
    feats["recent_activity_ratio"] = feats["events_last_7d"] / (feats["n_events"] / weeks_observed)

    feats["paid_share"] = (ev["level"].astype(str) == "paid").groupby(ev["userId"], observed=True).mean()
    last = ev.groupby("userId", observed=True).tail(1).set_index("userId")
    feats["is_paid_now"] = (last["level"].astype(str) == "paid").astype(int)

    if "status" in ev.columns:
        feats["http_404_rate"] = (ev["status"] == 404).groupby(ev["userId"], observed=True).mean()
    if "gender" in ev.columns:
        feats["is_male"] = (last["gender"].astype(str) == "M").astype(int)
    if "registration" in ev.columns:
        tenure = (reference_ts - last["registration"]).dt.total_seconds() / 86_400
        feats["tenure_days"] = tenure
    if "length" in ev.columns:
        songs = ev[ev["page"].astype(str) == "NextSong"]
        feats["listening_hours"] = (
            songs.groupby("userId", observed=True)["length"].sum().reindex(feats.index).fillna(0) / 3600
        )

    feats = feats.drop(columns=["first_event", "last_event"])
    feats.index = feats.index.astype(str)
    feats.index.name = "userId"
    return feats.replace([np.inf, -np.inf], np.nan).astype("float64")


# --------------------------------------------------------------------------- #
# Aggregations for charts
# --------------------------------------------------------------------------- #
def daily_activity(events: pd.DataFrame) -> pd.DataFrame:
    """Events and distinct active users per calendar day."""
    if events.empty:
        return pd.DataFrame(columns=["day", "events", "active_users"])
    day = events["time"].dt.normalize().rename("day")
    out = events.groupby(day).agg(events=("userId", "size"), active_users=("userId", "nunique"))
    return out.reset_index()


def page_distribution(events: pd.DataFrame) -> pd.DataFrame:
    """Event count and share per page, most frequent first."""
    counts = events["page"].astype(str).value_counts()
    out = counts.rename_axis("page").reset_index(name="events")
    out["share"] = out["events"] / max(int(out["events"].sum()), 1)
    return out


def churn_rate_by(features: pd.DataFrame, labels: pd.DataFrame, column: str, bins: int = 5) -> pd.DataFrame:
    """Churn rate by quantile bucket of a numeric feature (or by value if binary)."""
    joined = features[[column]].join(labels.set_index("userId")["churn"], how="inner").dropna()
    if joined.empty:
        return pd.DataFrame(columns=["bucket", "users", "churn_rate"])
    if joined[column].nunique() <= 2:
        bucket = joined[column].astype(int).astype(str)
    else:
        bucket = pd.qcut(joined[column], q=bins, duplicates="drop").astype(str)
    out = joined.groupby(bucket, sort=False).agg(users=("churn", "size"), churn_rate=("churn", "mean"))
    out = out.rename_axis("bucket").reset_index()
    order = joined.groupby(bucket, sort=False)[column].min().reindex(out["bucket"]).to_numpy()
    return out.iloc[np.argsort(order, kind="stable")].reset_index(drop=True)


def user_timeline(events: pd.DataFrame, user_id: str) -> pd.DataFrame:
    """Daily page counts for one user (long format: day, page, events)."""
    ev = events[events["userId"].astype(str) == str(user_id)]
    if ev.empty:
        return pd.DataFrame(columns=["day", "page", "events"])
    out = ev.groupby([ev["time"].dt.normalize().rename("day"), ev["page"].astype(str)]).size()
    return out.rename("events").reset_index()

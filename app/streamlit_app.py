"""Churn Explorer — Streamlit front-end for the churn_app package.

Run locally:   uv run streamlit run app/streamlit_app.py
Run in Docker: docker run -p 8501:8501 ghcr.io/martinoor/datascience:latest
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from churn_app import __version__
from churn_app.data import (
    CHURN_PAGE,
    EventFilter,
    SchemaError,
    churn_labels,
    filter_events,
    filter_options,
    load_events,
    normalize_events,
)
from churn_app.features import (
    build_user_features,
    churn_rate_by,
    daily_activity,
    page_distribution,
    user_timeline,
)
from churn_app.model import MODEL_LABELS, score_users, train_churn_model
from churn_app.sample_data import generate_events

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
KAGGLE_URL = "https://www.kaggle.com/competitions/churn-prediction-25-26"

st.set_page_config(page_title="Churn Explorer", page_icon="📉", layout="wide")


# --------------------------------------------------------------------------- #
# Cached computations. Arguments prefixed with "_" are not hashed; the explicit
# string keys identify the content instead, which avoids hashing large frames.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Generating synthetic events…", max_entries=4)
def load_sample(n_users: int, seed: int) -> pd.DataFrame:
    return normalize_events(generate_events(n_users=n_users, seed=seed))


@st.cache_data(show_spinner="Reading file…", max_entries=2)
def load_file(path: str, mtime: float, start: dt.date | None, end: dt.date | None, fraction: float) -> pd.DataFrame:
    del mtime  # only part of the cache key, so edited files are re-read
    return load_events(path, start=start, end=end, user_fraction=fraction)


@st.cache_data(show_spinner="Reading upload…", max_entries=2)
def load_upload(content: bytes, name: str, fraction: float) -> pd.DataFrame:
    return load_events(content, fmt=Path(name).suffix, user_fraction=fraction)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_filter(_events: pd.DataFrame, data_key: str, flt: EventFilter) -> pd.DataFrame:
    return filter_events(_events, flt)


@st.cache_data(show_spinner="Building user features…", max_entries=8)
def cached_features(_events: pd.DataFrame, key: str, reference: dt.date | None) -> pd.DataFrame:
    return build_user_features(_events, reference)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_labels(_events: pd.DataFrame, data_key: str, cutoff: dt.date | None, horizon: int) -> pd.DataFrame:
    return churn_labels(_events, cutoff, horizon)


# --------------------------------------------------------------------------- #
# Sidebar: data source
# --------------------------------------------------------------------------- #
def sidebar_data() -> tuple[pd.DataFrame, str, str] | None:
    st.sidebar.header("1 · Data")
    local_files = sorted(p for ext in ("*.parquet", "*.csv") for p in DATA_DIR.glob(ext)) if DATA_DIR.is_dir() else []
    sources = ["Synthetic sample", "Upload a file"]
    if local_files:
        sources.insert(0, f"File in {DATA_DIR}/")
    source = st.sidebar.radio("Source", sources, help=f"Mount real Kaggle files into `{DATA_DIR}/` to see them here.")

    if source == "Synthetic sample":
        n_users = st.sidebar.slider("Users", 100, 2_000, 400, step=100)
        seed = st.sidebar.number_input("Seed", 0, 10_000, 7)
        return load_sample(n_users, int(seed)), f"sample:{n_users}:{seed}", "synthetic sample"

    fraction = (
        st.sidebar.slider(
            "User sample (%)",
            1,
            100,
            100,
            help="Deterministic hash-based sample of users — use it for the full 17M-row training file.",
        )
        / 100
    )

    if source == "Upload a file":
        upload = st.sidebar.file_uploader("Parquet or CSV event log", type=["parquet", "csv"])
        if upload is None:
            return None
        content = upload.getvalue()
        key = f"upload:{hashlib.sha256(content).hexdigest()[:16]}:{fraction}"
        return load_upload(content, upload.name, fraction), key, upload.name

    path = st.sidebar.selectbox("File", local_files, format_func=lambda p: p.name)
    limit_dates = st.sidebar.checkbox("Only load a date range", value=False)
    start = end = None
    if limit_dates:
        start = st.sidebar.date_input("From", dt.date(2018, 10, 1))
        end = st.sidebar.date_input("To", dt.date(2018, 11, 20))
    mtime = path.stat().st_mtime
    key = f"file:{path}:{mtime}:{start}:{end}:{fraction}"
    return load_file(str(path), mtime, start, end, fraction), key, path.name


def sidebar_filters(events: pd.DataFrame) -> EventFilter:
    st.sidebar.header("2 · Filters")
    opts = filter_options(events)
    min_d, max_d = opts["min_date"], opts["max_date"]
    picked = st.sidebar.date_input("Event dates", (min_d, max_d), min_value=min_d, max_value=max_d)
    start, end = picked if isinstance(picked, tuple) and len(picked) == 2 else (min_d, max_d)

    def multi(label: str, column: str) -> tuple[str, ...]:
        values = opts[column]
        if not values:
            return ()
        return tuple(st.sidebar.multiselect(label, values, placeholder="All"))

    return EventFilter(
        start=start,
        end=end,
        levels=multi("Subscription level", "level"),
        genders=multi("Gender", "gender"),
        devices=multi("Device", "device"),
        states=multi("State", "state"),
        pages=multi("Pages", "page"),
        exclude_leakage=st.sidebar.toggle(
            "Hide cancellation events", value=False, help="Drop `Cancel` / `Cancellation Confirmation` rows."
        ),
    )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_overview(events: pd.DataFrame, labels_ever: pd.DataFrame) -> None:
    users = events["userId"].nunique()
    churned = int(labels_ever.loc[labels_ever["userId"].isin(events["userId"].unique()), "churn"].sum())
    c = st.columns(5)
    c[0].metric("Events", f"{len(events):,}")
    c[1].metric("Users", f"{users:,}")
    c[2].metric("Sessions", f"{events['sessionId'].nunique():,}")
    c[3].metric("Users who cancelled", f"{churned:,} ({churned / max(users, 1):.1%})")
    c[4].metric("Days covered", f"{(events['time'].max() - events['time'].min()).days + 1}")

    left, right = st.columns([3, 2])
    daily = daily_activity(events)
    long = daily.melt(id_vars="day", var_name="metric", value_name="count")
    long["metric"] = long["metric"].map({"events": "Events", "active_users": "Active users"})
    fig = px.line(long, x="day", y="count", facet_row="metric", title="Daily activity", height=420)
    fig.update_yaxes(matches=None, title_text="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(xaxis_title="", showlegend=False)
    left.plotly_chart(fig, width="stretch")

    pages = page_distribution(events)
    fig = px.bar(pages, x="events", y="page", orientation="h", title="Events by page", log_x=True)
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="events (log scale)", yaxis_title="")
    right.plotly_chart(fig, width="stretch")

    cancel_days = events.loc[events["page"].astype(str) == CHURN_PAGE, "time"].dt.normalize().value_counts()
    if not cancel_days.empty:
        fig = px.bar(
            cancel_days.sort_index().rename_axis("day").reset_index(name="cancellations"),
            x="day",
            y="cancellations",
            title="Cancellations per day",
        )
        st.plotly_chart(fig, width="stretch")


def tab_segments(events: pd.DataFrame, features: pd.DataFrame, labels_ever: pd.DataFrame) -> None:
    st.caption("User-level churn uses the *ever cancelled* label on the loaded data, before filters.")
    cols = st.columns(3)
    for col, column, title in zip(
        cols, ("level", "device", "state"), ("Subscription level", "Device", "State"), strict=True
    ):
        if column not in events.columns:
            continue
        last = events.groupby("userId", observed=True)[column].last().astype(str).rename(column)
        seg = last.to_frame().join(labels_ever.set_index("userId")["churn"], how="inner")
        agg = seg.groupby(column).agg(users=("churn", "size"), churn_rate=("churn", "mean")).reset_index()
        fig = px.bar(agg, x=column, y="churn_rate", hover_data=["users"], title=f"Churn rate by {title.lower()}")
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="")
        col.plotly_chart(fig, width="stretch")

    if features.empty:
        st.info("No users left after filtering.")
        return
    numeric = [c for c in features.columns if features[c].nunique() > 1]
    default = numeric.index("days_since_last_event") if "days_since_last_event" in numeric else 0
    feature = st.selectbox("Churn rate by quantile of…", numeric, index=default)
    buckets = churn_rate_by(features, labels_ever, feature)
    fig = px.bar(buckets, x="bucket", y="churn_rate", hover_data=["users"], title=f"Churn rate by {feature}")
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title=feature, yaxis_title="churn rate")
    st.plotly_chart(fig, width="stretch")
    if feature == "days_since_last_event":
        st.warning(
            "Recency looks extremely predictive here, but with the *ever cancelled* label churned users simply "
            "stop generating events. Use the horizon label in **Churn model** for an honest estimate."
        )


def tab_user(events: pd.DataFrame, labels_ever: pd.DataFrame) -> None:
    counts = events["userId"].value_counts()
    if counts.empty:
        st.info("No users left after filtering.")
        return
    churned_ids = set(labels_ever.loc[labels_ever["churn"] == 1, "userId"])
    show = st.radio("Pick from", ["Most active users", "Users who cancelled"], horizontal=True)
    candidates = [u for u in counts.index if u in churned_ids] if show == "Users who cancelled" else list(counts.index)
    if not candidates:
        st.info("No matching users in the filtered data.")
        return
    user = st.selectbox("User ID", candidates[:500])
    ev = events[events["userId"] == user]
    c = st.columns(4)
    c[0].metric("Events", f"{len(ev):,}")
    c[1].metric("Sessions", ev["sessionId"].nunique())
    c[2].metric("Current level", str(ev["level"].iloc[-1]))
    c[3].metric("Cancelled", "yes" if user in churned_ids else "no")
    tl = user_timeline(events, user)
    fig = px.bar(tl, x="day", y="events", color="page", title=f"Daily events for user {user}")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(ev.tail(50).iloc[::-1], width="stretch", hide_index=True)


def tab_model(all_events: pd.DataFrame, events: pd.DataFrame, data_key: str) -> None:
    opts = filter_options(all_events)
    st.markdown(
        "Trains a model on user-level features and tunes the decision threshold for **balanced accuracy**, "
        "the competition metric. Users are split into train / validation at random, stratified by label."
    )
    with st.form("model_form"):
        c = st.columns(4)
        mode = c[0].radio(
            "Label",
            ["Horizon (competition)", "Ever cancelled"],
            help="Horizon: features before the cutoff, label = cancels within the next N days. "
            "Ever: label = cancels at any time (the original project's proxy; leaks through recency).",
        )
        default_cutoff = max(opts["min_date"], opts["max_date"] - dt.timedelta(days=10))
        cutoff = c[1].date_input("Cutoff", default_cutoff, min_value=opts["min_date"], max_value=opts["max_date"])
        horizon = c[1].number_input("Horizon (days)", 1, 30, 10)
        model_type = c[2].selectbox("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get)
        val_size = c[2].slider("Validation share", 0.1, 0.5, 0.25, 0.05)
        seed = c[3].number_input("Random seed", 0, 10_000, 42)
        submitted = st.form_submit_button("Train model", type="primary")

    if submitted:
        horizon_mode = mode.startswith("Horizon")
        ref = cutoff if horizon_mode else None
        labels = cached_labels(all_events, data_key, ref, int(horizon))
        features = cached_features(events, data_key, ref)
        try:
            with st.spinner("Training…"):
                result = train_churn_model(features, labels, model_type, val_size, int(seed))
        except ValueError as exc:
            st.error(f"Cannot train: {exc}")
            return
        st.session_state["trained_model"] = (data_key, result, features, mode, cutoff, horizon)

    if st.session_state.get("trained_model", (None,))[0] != data_key:
        st.info("Choose settings and press **Train model**.")
        return

    _, result, features, mode, cutoff, horizon = st.session_state["trained_model"]
    m = result.metrics
    st.subheader(f"{mode} · cutoff {cutoff} · {horizon}-day horizon" if mode.startswith("Horizon") else mode)
    c = st.columns(6)
    c[0].metric("Balanced accuracy", f"{m['balanced_accuracy']:.3f}")
    c[1].metric("ROC AUC", f"{m['roc_auc']:.3f}")
    c[2].metric("Recall", f"{m['recall']:.1%}")
    c[3].metric("Precision", f"{m['precision']:.1%}")
    c[4].metric("Users flagged", f"{m['flagged_share']:.1%}")
    c[5].metric("Threshold", f"{result.threshold:.2f}")
    st.caption(
        f"{int(m['n_train']):,} training / {int(m['n_val']):,} validation users · "
        f"base churn rate {m['val_churn_rate']:.1%}"
    )

    left, right = st.columns(2)
    imp = result.importance.head(15)
    fig = px.bar(
        imp,
        x="importance",
        y="feature",
        error_x="std",
        orientation="h",
        title="Permutation importance (drop in validation AUC)",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="")
    left.plotly_chart(fig, width="stretch")

    val = result.validation.assign(outcome=lambda d: d["churn"].map({0: "retained", 1: "churned"}))
    fig = px.histogram(
        val, x="proba", color="outcome", nbins=30, barmode="overlay", title="Validation churn probability"
    )
    fig.add_vline(x=result.threshold, line_dash="dash", annotation_text="threshold")
    right.plotly_chart(fig, width="stretch")
    right.dataframe(result.confusion, width="stretch")

    scores = score_users(result, features)
    st.markdown("**All users, ranked by churn risk**")
    st.caption("Scores for training users are in-sample; use the validation metrics above to judge the model.")
    st.dataframe(scores.head(200), width="stretch")
    st.download_button("Download scores (CSV)", scores.to_csv().encode(), "churn_scores.csv", "text/csv")


def tab_data(events: pd.DataFrame, source_name: str) -> None:
    st.markdown(f"**Source:** `{source_name}` · {len(events):,} rows after filters")
    st.dataframe(events.head(500), width="stretch", hide_index=True)
    profile = pd.DataFrame(
        {
            "dtype": events.dtypes.astype(str),
            "null share": events.isna().mean(),
            "distinct": events.nunique(),
        }
    )
    st.markdown("**Column profile**")
    st.dataframe(profile, width="stretch")
    if len(events) <= 1_000_000:
        st.download_button(
            "Download filtered events (CSV)", events.to_csv(index=False).encode(), "events_filtered.csv", "text/csv"
        )
    else:
        st.caption("Filtered data is over 1M rows; narrow the filters to enable CSV download.")


def tab_about() -> None:
    st.markdown(
        f"""
### About
Interactive companion to the **[Kaggle churn-prediction-25-26]({KAGGLE_URL})** project: predict whether a
streaming-service user visits *Cancellation Confirmation* in the 10 days after the observation window.

**Data.** The competition files can't be redistributed, so the app ships with a *deterministic synthetic*
generator that reproduces the schema (19 columns) and the page mix. To use the real data, download
`train.parquet` from Kaggle and either upload it or mount its folder at `{DATA_DIR}/`
(`docker run -v $PWD/churn-prediction-25-26:/data …`). First/last names are dropped when files are loaded.

**Leakage controls.** Features never use `Cancel` / `Cancellation Confirmation` pages or `Cancelled` auth
events, and in horizon mode only events strictly before the cutoff are used.

**Model.** Fast CPU models from scikit-learn, for interactive use. The Transformer / XGBoost research
pipelines live in `final_experiments/` in the repository.

Version `{__version__}`.
"""
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📉 Churn Explorer")
    st.caption("Explore streaming-service event logs, filter them, and train a churn model.")

    try:
        loaded = sidebar_data()
    except (SchemaError, ValueError, OSError) as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()
    if loaded is None:
        st.info("Upload a `.parquet` or `.csv` event log in the sidebar to begin.")
        tab_about()
        st.stop()

    all_events, data_key, source_name = loaded
    if all_events.empty:
        st.warning("The selected data contains no events.")
        st.stop()

    flt = sidebar_filters(all_events)
    events = cached_filter(all_events, data_key, flt)
    st.sidebar.caption(f"{len(events):,} of {len(all_events):,} events after filters")
    if events.empty:
        st.warning("No events match the current filters.")
        st.stop()

    labels_ever = cached_labels(all_events, data_key, None, 10)
    features = cached_features(events, f"{data_key}:{flt}", None)

    tabs = st.tabs(["Overview", "Segments", "User drill-down", "Churn model", "Data", "About"])
    with tabs[0]:
        tab_overview(events, labels_ever)
    with tabs[1]:
        tab_segments(events, features, labels_ever)
    with tabs[2]:
        tab_user(events, labels_ever)
    with tabs[3]:
        tab_model(all_events, events, f"{data_key}:{flt}")
    with tabs[4]:
        tab_data(events, source_name)
    with tabs[5]:
        tab_about()


main()

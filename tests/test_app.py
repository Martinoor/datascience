"""End-to-end smoke tests that execute the Streamlit script headlessly."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
pytestmark = pytest.mark.slow


@pytest.fixture
def app(monkeypatch, tmp_path) -> AppTest:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))  # empty: sample data is the default source
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_app_renders_with_sample_data(app: AppTest):
    assert not app.exception
    assert app.title[0].value.startswith("📉 Churn Explorer")
    labels = [m.label for m in app.metric]
    assert {"Events", "Users", "Sessions"} <= set(labels)
    assert len(app.tabs) == 6


def test_app_filters_update_metrics(app: AppTest):
    events_before = next(m.value for m in app.metric if m.label == "Events")
    level = next(w for w in app.sidebar.multiselect if w.label == "Subscription level")
    level.select("paid").run()
    assert not app.exception
    events_after = next(m.value for m in app.metric if m.label == "Events")
    assert int(events_after.replace(",", "")) < int(events_before.replace(",", ""))


def test_app_trains_model(app: AppTest):
    train = next(b for b in app.button if b.label == "Train model")
    train.click().run()
    assert not app.exception
    labels = {m.label for m in app.metric}
    assert {"Balanced accuracy", "ROC AUC", "Threshold"} <= labels


def test_app_lists_files_in_data_dir(monkeypatch, tmp_path):
    from churn_app.sample_data import generate_events

    generate_events(60, seed=5).to_parquet(tmp_path / "train.parquet")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert at.sidebar.radio[0].value.startswith("File in")
    users = next(m.value for m in at.metric if m.label == "Users")
    assert users == "60"

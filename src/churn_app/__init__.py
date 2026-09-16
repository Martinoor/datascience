"""Churn explorer: data loading, filtering, features and modelling for the
Kaggle *churn-prediction-25-26* streaming-service event logs."""

from churn_app.data import (
    CHURN_PAGE,
    EventFilter,
    SchemaError,
    churn_labels,
    filter_events,
    filter_options,
    load_events,
)

__all__ = [
    "CHURN_PAGE",
    "EventFilter",
    "SchemaError",
    "churn_labels",
    "filter_events",
    "filter_options",
    "load_events",
]

__version__ = "1.0.0"

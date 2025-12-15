# Target / Label Notes

## 1) What we predict

Churn is defined as the user visiting the page:

- `page == "Cancellation Confirmation"`

The training target is binary.

## 2) Mismatch with the competition description

The Kaggle description mentions:

> "…churn in the window of 10 days that follows the given observations (ie after 2018-11-20)"

However, the parquet files used in this repo have event timestamps that end at `2018-11-20`. With that data alone, we cannot directly build a label strictly based on “(2018-11-20, 2018-11-30]”.

Therefore, this project uses one of these label modes (see `FeatureBuildConfig` in `src/churn_pipeline/dataset_builder.py`):

- `label_mode="ever"` (default): `label=1` if the user ever has a `Cancellation Confirmation` event in their available history.
- `label_mode="horizon"` (rolling cutoffs): with a chosen `cutoff_time`, `label=1` if the churn event happens in $(cutoff, cutoff + horizon\_days]$.

## 3) Leakage controls

To avoid directly leaking the label into features, the feature builder excludes:

- `page in {"Cancellation Confirmation", "Cancel"}` (configurable)
- `auth == "Cancelled"` (configurable)
- optionally also drops the last `k` events per user (`drop_last_k_events`) as an additional near-end leakage guard

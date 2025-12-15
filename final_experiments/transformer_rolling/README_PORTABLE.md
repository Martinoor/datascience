# Portable experiment: transformer_rolling

This directory is intended to be **self-contained** so you can copy it elsewhere and still run training/prediction notebooks.

## What’s included

- `src/churn_pipeline/`: model + dataset code used by the notebooks
- `scripts/`: dataset building CLI and helper scripts
- `churn-prediction-25-26/`: raw competition files (train/test parquet + example submission)
- `data/`: processed datasets (if already generated)
- `artifacts/`, `submissions/`: model artifacts and submission csv outputs

## How to run

1. Open a terminal and set working directory to this folder.

   - `cd /path/to/transformer_rolling`

2. Start Jupyter from here (recommended) so relative paths are stable.

3. Run the notebook:

- `transformer_rolling_train_predict.ipynb`
- `resnet_transformer_rolling_train_predict.ipynb`

The notebooks auto-detect `ROOT` by searching upward for both `src/churn_pipeline` and `churn-prediction-25-26`.

## Notes

- If you copy this folder to a new location, existing `.cache/` entries may not be reused because cache keys include the absolute path of the parquet files. This is expected; you can delete `.cache/` to force a clean rebuild.
- Kaggle submission requires `~/.kaggle/kaggle.json` and Kaggle CLI login.

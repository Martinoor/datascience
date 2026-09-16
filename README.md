# Churn Prediction (Kaggle Competition 25/26)

[![CI](https://github.com/Martinoor/datascience/actions/workflows/ci.yml/badge.svg)](https://github.com/Martinoor/datascience/actions/workflows/ci.yml)
[![Image](https://img.shields.io/badge/image-ghcr.io%2Fmartinoor%2Fchurn--explorer-2496ED?logo=docker&logoColor=white)](https://github.com/Martinoor/datascience/pkgs/container/churn-explorer)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)

A machine learning project for predicting user churn from streaming service logs. This project implements multiple approaches including **Transformer-based sequence models**, **XGBoost**, and **ensemble methods** to achieve competitive performance on the [Kaggle Churn Prediction Competition](https://www.kaggle.com/competitions/churn-prediction-25-26).

The repository has two parts:

1. **Churn Explorer**: a tested, containerised **Streamlit app** (`src/churn_app`, `app/`) to load, filter and explore the event logs and train a churn model interactively. See [Churn Explorer app](#-churn-explorer-app) below.
2. **Research pipelines**: the original notebooks, Transformer / XGBoost / ensemble experiments (`final_experiments/`, root `*.py`, `*.ipynb`). See [Problem statement](#-problem-statement) onwards.

---

## 📉 Churn Explorer app

### Run it

| How | Command | Then open |
|-----|---------|-----------|
| Pre-built image | `docker run --rm -p 8501:8501 ghcr.io/martinoor/churn-explorer:latest` | http://localhost:8501 |
| Docker Compose (build locally) | `docker compose up --build` | http://localhost:8501 |
| Local Python | `uv sync --locked && uv run streamlit run app/streamlit_app.py` | http://localhost:8501 |

Without any data, the app starts on a **deterministic synthetic sample** that has the same 19-column schema and page mix as the Kaggle files. To explore the real data, download `train.parquet` / `test.parquet` from the [competition page](https://www.kaggle.com/competitions/churn-prediction-25-26/data) (the data can't be redistributed) and choose one of:

- **Mount it**: `docker run --rm -p 8501:8501 -v "$PWD/churn-prediction-25-26:/data:ro" ghcr.io/martinoor/churn-explorer:latest`. Files in `/data` (or `$DATA_DIR` locally, default `./data`) appear under *Source → File in /data/*.
- **Upload it** from the sidebar (up to 2 GB).

The full training file has ~17.5M rows. Use the sidebar's **User sample (%)** and **date range** options: they are applied batch by batch while reading, so memory stays bounded.

### What it does

| Tab | Content |
|-----|---------|
| **Overview** | Events, users, sessions, cancellation rate; daily events / active users; page mix; cancellations per day |
| **Segments** | Churn rate by plan, device, state, and by quantile of any user-level feature |
| **User drill-down** | Per-user daily page timeline and raw events |
| **Churn model** | Gradient boosting or logistic regression on user features, **horizon** (competition) or *ever-cancelled* labels, threshold tuned for **balanced accuracy**, permutation importance, confusion matrix, downloadable risk scores |
| **Data** | Filtered rows, column profile, CSV export |

Sidebar filters (dates, plan, gender, device, state, pages, hide cancellation events) apply to every tab.

### Design notes

- **Label leakage.** Features never use `Cancel` / `Cancellation Confirmation` pages or `auth == "Cancelled"`. In horizon mode they only use events strictly before the cutoff, and a test checks that adding future events changes nothing. The *ever-cancelled* label (used by the research notebooks) is available, but the app warns that recency leaks the answer under it.
- **Privacy.** `firstName` / `lastName` are dropped when files are read.
- **Typing.** User IDs are always strings: with CSV input, one blank ID would otherwise turn `1749042` into `1749042.0`. There is a test for this.

### Project layout (app)

```
src/churn_app/
  data.py          # read (parquet/csv, path/bytes/upload), validate, normalise, EventFilter, churn_labels
  features.py      # leakage-safe user-level features + chart aggregations
  model.py         # training, balanced-accuracy threshold tuning, scoring
  sample_data.py   # deterministic synthetic generator (also a CLI)
app/streamlit_app.py   # UI only: caching, widgets, charts
tests/                 # unit tests + headless Streamlit AppTest end-to-end tests
Dockerfile, docker-compose.yml, .dockerignore
.github/workflows/ci.yml, .github/dependabot.yml, .pre-commit-config.yaml
pyproject.toml, uv.lock, .python-version, Makefile
```

### Development

```bash
uv sync --locked            # exact environment from uv.lock (Python 3.12, see .python-version)
make lint                   # lockfile check + ruff lint + ruff format --check
make test                   # pytest with branch coverage (fails under 85%)
make run                    # streamlit on :8501
make sample                 # write data/sample_events.parquet via the generator CLI
make docker-build docker-run
uv tool install pre-commit && pre-commit install   # same checks on every commit
```

The tests cover the data layer in detail: format inference, reading from paths, bytes and file-like uploads, missing-column errors, `ts` → `time` derivation, timezone handling, day-inclusive date ranges, deterministic user sampling that is identical across batch sizes, device/state parsing, every filter dimension and their AND/OR semantics, and the edges of the horizon label window. The app itself is exercised headlessly with `streamlit.testing.v1.AppTest`: it renders, filters, trains a model and picks up files in `DATA_DIR`.

### CI/CD and reproducibility

GitHub Actions ([`ci.yml`](.github/workflows/ci.yml)) runs on every push and pull request:

1. **Lint**: `uv lock --check`, then ruff lint and format checks.
2. **Test**: pytest with coverage on Python 3.12 and 3.13, installed with `uv sync --locked`. JUnit and coverage reports are uploaded as artifacts.
3. **Docker**: builds the image, generates data inside it, starts the container, waits for the health check, checks the HTTP endpoints and that it runs as the non-root user. On `main` and `v*.*.*` tags, it pushes a multi-arch (`linux/amd64`, `linux/arm64`) image with SBOM and provenance to `ghcr.io/martinoor/churn-explorer`, tagged `latest`, `sha-<short>` and semver. Pushing to Docker Hub as well is optional: set the repo variable `DOCKERHUB_USERNAME` and the secret `DOCKERHUB_TOKEN`.

Reproducibility guarantees:

- Python dependencies are pinned in `uv.lock` and installed with `--locked` / `--frozen` everywhere (local, CI, Docker).
- The Python version is pinned in `.python-version`; base images (`python:3.12-slim-bookworm`, `uv`) are pinned **by digest**.
- Model training, the train/validation split, user sampling and synthetic data are all seeded, and tests check that repeated runs give identical output.
- Dependabot opens weekly PRs for uv, Docker base images and GitHub Actions, so pins don't go stale.
- The runtime image contains only the locked virtualenv, the app and its config. It runs as UID 10001 with a `HEALTHCHECK`.

To reproduce a published image exactly, use its `sha-<commit>` tag or check out that commit and run `docker build .`.

---

## 🎯 Problem Statement

Predict whether users will churn (visit the `Cancellation Confirmation` page) within a 10-day window following the observation period (after `2018-11-20`).

- **Input**: User behavior event sequences from a streaming service
- **Output**: Binary classification (churn: 0/1)
- **Evaluation Metric**: Balanced Accuracy Score = (TPR + TNR) / 2

## 📊 Dataset Overview

| Split | Users | Churn Rate | Event Time Range |
|-------|-------|------------|------------------|
| Train | 19,140 | ~22.3% | 2018-10-01 ~ 2018-11-20 |
| Test | TBD | N/A | 2018-10-01 ~ 2018-11-20 |

---

## 🏗️ Project Architecture

```
py_kaggle/
│
├── 📁 churn-prediction-25-26/          # Raw Kaggle competition data
│   ├── train.parquet                   # Training data (user events)
│   ├── test.parquet                    # Test data (user events)
│   └── example_submission.csv          # Submission format example
│
├── 📁 final_experiments/               # Production-ready experiment pipelines
│   │
│   ├── 📁 transformer_rolling/         # Transformer with rolling window
│   │   ├── src/churn_pipeline/         # Core model & dataset modules
│   │   │   ├── dataset_builder.py      # Rolling window dataset construction
│   │   │   ├── transformer_user_day.py # Transformer model definition
│   │   │   ├── resnet_transformer_user_day.py  # ResNet-Transformer hybrid
│   │   │   └── xgb_features.py         # XGBoost feature extraction
│   │   ├── scripts/                    # CLI utilities
│   │   │   └── build_datasets.py       # Dataset building CLI
│   │   ├── transformer_rolling_train_predict.ipynb
│   │   ├── resnet_transformer_rolling_train_predict.ipynb
│   │   ├── data/processed/             # Cached processed datasets
│   │   ├── artifacts/                  # Trained model checkpoints
│   │   └── submissions/                # Generated submission files
│   │
│   ├── 📁 xgb_rolling/                 # XGBoost with sliding window
│   │   ├── run_rolling_xgb.py          # Main training script
│   │   ├── xgb_rolling_train_predict.ipynb
│   │   ├── data/processed/             # Cached processed datasets
│   │   ├── artifacts/                  # Trained model checkpoints
│   │   └── submissions/                # Generated submission files
│   │
│   ├── 📁 ensemble/                    # Model blending & stacking
│   │   ├── blend_xgb_transformer_balacc.py  # Logit-space blending
│   │   └── ensemble_rolling_train_predict.ipynb
│   │
│   ├── best_params.json                # Best hyperparameters found
│   ├── data_features.md                # Feature engineering documentation
│   └── target_analysis.md              # Label distribution analysis
│
├── 📁 runs/                            # Training run artifacts & logs
│   └── event_ensemble/                 # Seed ensemble experiment runs
│       └── <timestamp>_<config>/       # Individual run directories
│           └── run_meta.json           # Run configuration & metrics
│
├── 📁 feature_cache/                   # Cached feature computations
│
├── 📁 __pycache__/                     # Python bytecode cache
│
│
├── ──────────────────────────────────  # ═══ Core Pipeline Modules ═══
│
├── 🐍 feature_pipeline.py              # Feature engineering pipeline
│                                       # - Event-level features (time, session, etc.)
│                                       # - Categorical encodings
│                                       # - Sequence truncation & padding
│                                       # - Train/val/test dataset preparation
│
├── 🐍 transformer_model.py             # Transformer model architecture
│                                       # - ChurnTransformer class
│                                       # - Attention pooling
│                                       # - Focal loss support
│                                       # - Training loop with early stopping
│
├── 🐍 train_event_ensemble.py          # Seed ensemble training script
│                                       # - Multi-seed training for robustness
│                                       # - Probability averaging
│                                       # - Threshold optimization
│
├── 🐍 kaggle_submit.py                 # Kaggle submission helper
│                                       # - API integration
│                                       # - Score polling
│                                       # - Submission logging
│
├── 🐍 submission_utils.py              # Submission file utilities
│                                       # - Model fitting wrappers
│                                       # - CSV generation
│
│
├── ──────────────────────────────────  # ═══ Notebooks ═══
│
├── 📓 EDA_test.ipynb                   # Exploratory Data Analysis
├── 📓 feature_engineering.ipynb        # Feature engineering experiments
├── 📓 model_construction.ipynb         # Main model training notebook
├── 📓 classical_models.ipynb           # Traditional ML baselines
├── 📓 test.ipynb                       # Debugging & testing notebook
│
│
├── ──────────────────────────────────  # ═══ Documentation & Logs ═══
│
├── 📄 prompt.md                        # Tuning cheat sheet & guidelines
├── 📄 data_features.md                 # Feature documentation
├── 📄 tuning_log.csv                   # Hyperparameter tuning history
├── 📄 submission_log.csv               # Kaggle submission history
│
│
├── ──────────────────────────────────  # ═══ Outputs ═══
│
├── 📊 submission.csv                   # Latest submission file
├── 📊 submission_event_ensemble.csv    # Ensemble model submission
├── 🎨 training_loss.png                # Training curves visualization
└── 🏆 transformer_best.pt              # Best model checkpoint
```

---

## 🔧 Key Components

### Feature Engineering (`feature_pipeline.py`)

Extracts rich features from raw event sequences:

| Feature Category | Examples |
|-----------------|----------|
| **Temporal** | `seconds_since_prev_event`, `hour_sin/cos`, `dow_sin/cos` |
| **Session** | Event index, session duration, session progress |
| **Subscription** | Level changes, upgrade/downgrade counts |
| **Behavior** | Page visit patterns, 404 error ratio |
| **Content** | Distinct songs/artists, listening concentration |
| **Categorical** | Page ID, device type, metro area, state |

### Model Architectures

1. **Transformer** (`transformer_model.py`)
   - Attention-based sequence encoder
   - Configurable pooling (mean / attention)
   - Focal loss for class imbalance
   - Cosine annealing LR scheduler

2. **XGBoost Rolling** (`final_experiments/xgb_rolling/`)
   - Sliding window approach
   - Multi-cutoff snapshot concatenation
   - Gradient boosted trees

3. **Ensemble** (`final_experiments/ensemble/`)
   - Logit-space blending of Transformer + XGBoost
   - Grid search for optimal blend weights

### Training Strategies

- **Rolling Window**: Train on multiple cutoff dates to simulate temporal validation
- **Seed Ensemble**: Average predictions across multiple random seeds
- **Threshold Optimization**: Grid search for balanced accuracy

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install pandas numpy torch scikit-learn xgboost tqdm kaggle matplotlib

# Setup Kaggle API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Training

**Option 1: Interactive Notebook**
```bash
# Open and run cell-by-cell
jupyter notebook model_construction.ipynb
```

**Option 2: Ensemble Training Script**
```bash
python train_event_ensemble.py
```

**Option 3: XGBoost Rolling**
```bash
cd final_experiments/xgb_rolling
python run_rolling_xgb.py --xgb-device cuda
```

### Submission

Submissions are automatically tracked in `submission_log.csv`. To manually submit:

```python
from kaggle_submit import submit_and_track

submit_and_track("submission.csv", "churn-prediction-25-26", "run-note")
```

---

## 📈 Tuning Guide

See [prompt.md](prompt.md) for detailed tuning strategies:

| Issue | Solution |
|-------|----------|
| **Overfitting** | ↑ dropout (0.18-0.22), ↑ weight_decay (2e-3), ↓ max_seq_len |
| **Underfitting** | ↑ num_layers, ↑ dim_feedforward, ↑ epochs |
| **Low Recall** | ↑ pos_weight (×1.1-1.3), ↑ focal_gamma (+0.2) |
| **Low Precision** | ↓ pos_weight (×0.8-0.9), ↑ threshold |

---

## 📁 Directory Reference

| Directory | Purpose |
|-----------|---------|
| `churn-prediction-25-26/` | Raw competition data (parquet files) |
| `final_experiments/` | Production experiment pipelines |
| `runs/` | Training artifacts and metrics |
| `feature_cache/` | Cached feature computations |
| `*.ipynb` | Interactive notebooks for development |
| `*.py` | Reusable Python modules |

---

## 📝 Logs & Tracking

- **`tuning_log.csv`**: Hyperparameters, validation metrics, Kaggle scores
- **`submission_log.csv`**: Submission history with timestamps and scores
- **`runs/<experiment>/`**: Per-run artifacts (configs, plots, checkpoints)

---

## 🏆 Best Results

Check `final_experiments/best_params.json` for the current best configuration and `submission_log.csv` for historical Kaggle scores.

---

## License

This project is for educational purposes as part of the Kaggle competition.

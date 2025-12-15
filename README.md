# Churn Prediction (Kaggle Competition 25/26)

A machine learning project for predicting user churn from streaming service logs. This project implements multiple approaches including **Transformer-based sequence models**, **XGBoost**, and **ensemble methods** to achieve competitive performance on the [Kaggle Churn Prediction Competition](https://www.kaggle.com/competitions/churn-prediction-25-26).

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

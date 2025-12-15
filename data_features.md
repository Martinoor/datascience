# Transformer Model Features and Metrics Notes

## 1. Problem, Data, and Target Overview

- Task (per the Kaggle data): use each user's behavior sequence within the observation window to predict whether the user will churn (i.e., whether they visit the page `Cancellation Confirmation`).
- Sample definition:
  - Data comes from `train.parquet` / `test.parquet`; each row is one user behavior event.
  - Sequence length varies widely by user (from a few events to 10k+ events).
- Event time range (verified from parquet statistics):
  - `train.parquet`: `2018-10-01` ~ `2018-11-20`
  - `test.parquet`: `2018-10-01` ~ `2018-11-20`
- Note on the “10-day window after 2018-11-20”:
  - The project description mentions “a 10-day window after 2018-11-20”, but the released `train/test.parquet` events themselves end at `2018-11-20`, so **we cannot directly construct** a label for “10 days after 2018-11-20” from the provided data.
  - In practice (also matching the Kaggle data shape: `test` has no cancellation events, `train` has cancellation events), this project defines churn as “whether the user ever has `Cancellation Confirmation`”, and trains/submits accordingly.
- Label (churn) construction (current pipeline):
  - Aggregate `train.parquet` by `userId`. If any event in the user's full history has `page == "Cancellation Confirmation"`, set `churn = 1`; otherwise `0`.
  - For users who eventually cancel, `Cancellation Confirmation` usually appears at the very end (last one or a few events). This information must not be used as a feature (label leakage).
- Current label statistics (train):
  - Users: 19,140
  - churn=1: 4,271 (~22.3%)
  - churn=0: 14,869 (~77.7%)
- Evaluation metric (official Kaggle): Balanced Accuracy Score
  - `balanced_acc = (TPR + TNR) / 2`
- Prediction output (submission file): `id,target`, where `target` is 0/1 (binary label).
  - Evidence: `churn-prediction-25-26/example_submission.csv` also uses 0/1 labels.

---

!!!!The only official project description:
 Here's the kaggle link for the competition : https://www.kaggle.com/competitions/churn-prediction-25-26
The competition is to be performed in groups of two. You'll have a report of 4 pages to submit by december 14th, presenting the methods you tested and used. For the defense you'll get 8 minutes of presentations + 7 minutes of questions, including on question on the labs, that may involve writing a code snippet.
kaggle.com
Churn prediction 25/26
Predict churn prediction from streaming service logs
9:19
The goal of the competition is to predict whether or not some users (whose user ids are in the test file) will churn in the window of 10 days that follows the given observations (ie after "2018-11-20"). We consider that a user churns when they visit the page 'Cancellation Confirmation' (edited)
9:21
This is not a trivial challenge ; to get a decent score, we strongly advise you to start working on it right away.

## 2. Are the Current Features Sufficient? (based on `feature_pipeline.py`)

### 2.1 What is already covered
The current features (per-event numeric features + categorical embeddings for the Transformer) already cover most major information dimensions, including:

- The behavior sequence itself: `page_id` / `prev_page_id` (sequence patterns learned via page embeddings)
- Time and cadence: `seconds_since_prev_event`, `hour_sin/cos`, `dow_sin/cos`
- Session structure: event index within session, session duration, progress, etc.
- Subscription state and changes: `level`, cumulative upgrade/downgrade counts, time/events since last level change
- Quality/anomalies: historical 404 ratio (`error occur`)
- Content consumption and concentration: distinct song/artist, top1/top3 fractions and counts
- Skipping behavior: skip flags and skip ratios (cumulative + rolling window)
- User context: device / metro / state (categorical embeddings)

### 2.2 Potential gaps (recommended additions)
If you want to keep pushing for gains, prioritize information that is more directly related to churn and does not introduce leakage:

- Stronger “activity/decay” signals: active days (daily granularity), number of sessions in the last N days, days since last activity before cutoff (recency)
- Stronger “behavior shift” signals: change rates of key-page ratios over the last N events / last N days (e.g., Help/Settings/Error/Logout)
- Expand the page set: currently rolling/count features are built only for `KEY_PAGES`; try adding pages that may be more churn-related (e.g., Thumbs Down, Roll Advert, Add to Playlist), after checking whether they are common in the data

### 2.3 Should we prune features? (pruning/speed suggestions)
I recommend thinking about “pruning” in two buckets: pure redundant computation (should remove), vs. potentially useful but needs ablation.

- Clear redundant computation (already handled): older `feature_engineer()` versions generated full page one-hot columns, but `prepare_datasets()` later dropped all those columns (`drop_cols = set(page_categories)`); this was pure waste and has already been removed.
- Candidate for ablation: `get_song_stats_fast()` and some rolling features can be expensive. If training time or cache size becomes an issue, consider making them “toggleable feature groups” and decide based on online score / local balanced accuracy comparisons.

## 3. Local validation vs. Kaggle score: the issue is mainly metric misalignment

Your local reporting focuses on `val_auc` plus “pick threshold by F1”, while Kaggle evaluates **Balanced Accuracy**. It is very common to see “local looks good, online score differs” under this mismatch.

### 3.1 Does the label match Kaggle?
- Verified from the data: both `train/test` event times end at `2018-11-20`; `test` has 0 rows of `Cancellation Confirmation`, while `train` has 4,271 rows (corresponding to 4,271 users).
- This strongly suggests the churn label is essentially “whether the user ever has `Cancellation Confirmation`”, which matches your current `feature_pipeline.prepare_datasets()` label construction.

### 3.2 Why is validation so different from Kaggle?
Two alignment issues stack up:

- Metric mismatch: locally you look at AUC (threshold-free), but Kaggle uses Balanced Accuracy (threshold-dependent).
- Threshold objective mismatch: you choose threshold by F1, but Kaggle rewards Balanced Accuracy; for the same probability outputs, their optimal thresholds are often different.

### 3.3 Recommended local validation (without changing code yet)
- Log at least two numbers: `val_balanced_accuracy` (with the same 0/1 outputs as submission) + `val_auc` (to monitor ranking quality).
- Choose threshold by maximizing `balanced_accuracy` (scan thresholds), not by F1.
- If you want validation to better match the online distribution, move from random user split toward a more time-consistent holdout (e.g., a cutoff closer to `2018-11-20` or time-based holdout), then check whether online scores become more stable.

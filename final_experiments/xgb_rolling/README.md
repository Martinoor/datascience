# XGB Rolling Cutoff (Sliding Window) Training

This directory implements an XGBoost training workflow **isolated from the rest of the project**:

- For a sequence of `cutoff_time` values (e.g. 2018-10-20 → 2018-11-10, every 2 days), repeatedly build dataset snapshots:
  - Features: only use events strictly before the cutoff (enforced by `build_datasets_cli`)
  - Labels: `label_mode="horizon"` + `horizon_days=10`, i.e. **CC occurs in $(cutoff, cutoff+10\,days] \Rightarrow 1$, else 0**
- Concatenate samples across cutoffs (the same user at different cutoffs is treated as different samples; ids become `userId|YYYY-MM-DD`)
- Train XGB on the concatenated train/val and generate a test submission using a fixed cutoff (default: 2018-11-10)

## Run

From the project root:

```bash
python experiments/xgb_rolling/run_rolling_xgb.py \
  --cutoff-start 2018-10-20 \
  --cutoff-end 2018-11-10 \
  --cutoff-freq 2D \
  --submission-cutoff 2018-11-10 \
  --horizon-days 10 \
  --lookback-days 51 \
  --xgb-device cuda
```

If you do not have CUDA / GPU:

```bash
python experiments/xgb_rolling/run_rolling_xgb.py --xgb-device cpu
```

Outputs:

- Models and training metadata: `experiments/xgb_rolling/artifacts/<run_name>/...`
- Submission file: `experiments/xgb_rolling/submissions/<run_name>.csv`

## Notes

- This rolling-window approach is “multi-cutoff snapshot concatenation”, not rewriting the parquet files.
- The train/val split is still controlled by the hash split in `build_datasets_cli`.
  Because we only append `|cutoff` to `userId` after splitting, the same user will not leak across train/val.

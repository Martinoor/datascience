# Tuning Cheat Sheet (churn-prediction-25-26)

- Entry: run `model_construction.ipynb` cell-by-cell. `SUBMIT_TO_KAGGLE` controls auto-submission; `run_note` is automatically tagged with epoch/threshold.
- Logs: the final cell writes `tuning_log.csv` (hyperparams + val metrics + Kaggle score + next-step suggestion) and `submission_log.csv` (submission details). After each run, first check whether `tuning_log.csv` shows `improved_vs_prev=True`.
- Environment: keep `prepare_datasets` arguments unchanged; requires `~/.kaggle/kaggle.json` (chmod 600) and the `kaggle` package installed.
- What to watch: training/validation loss curves, val AUC/F1/ACC, and Kaggle public score. Threshold is grid-searched automatically; `best_threshold` updates based on validation.

## Quick Tuning Logic
- Mild overfitting (val loss clearly above train): raise `dropout` to 0.18~0.22, increase `weight_decay` to 2e-3, or reduce `max_seq_len` to 400.
- Underfitting (train/val both high and close): add capacity (`num_layers` → 5 or `dim_feedforward` → 896), or slightly increase `epochs` + `warmup_epochs`.
- Low recall, high precision: `pos_weight` *1.1~1.3, `focal_gamma` +0.2, and densify the threshold grid around 0.35~0.55.
- Low precision, high recall: `pos_weight` *0.8~0.9, `focal_gamma` -0.2, and shift threshold upward to 0.55~0.65.
- Already close to target (Kaggle > 0.65): small-step scan around current config for learning rate (5e-4, 7e-4), `max_seq_len` (400 vs 500), and fine-tune threshold.

## Suggested Next Experiments
1) Run once with the notebook default config; inspect `tuning_log.csv` and the curves.
2) If not overfitting and val AUC < 0.66: try `num_layers=5`, `dim_feedforward=896`, `dropout=0.18`, `lr=5e-4`.
3) If overfitting: keep layer count unchanged; try `dropout=0.22`, `weight_decay=2e-3`, `max_seq_len=400`.
4) If recall is insufficient: based on current config, set `pos_weight` *1.2, `focal_gamma=1.8`, and change threshold grid to 0.3~0.6 with step 0.01 (modify in the validation stage).

## Tips
- For each new config, always update `run_note` so logs are distinguishable.
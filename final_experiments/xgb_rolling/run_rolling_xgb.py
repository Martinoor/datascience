from __future__ import annotations
#%%
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from churn_pipeline.dataset_builder import build_datasets_cli  # noqa: E402

#%%
def _to_day(value: str | pd.Timestamp | np.datetime64) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]")
    return np.datetime64(pd.Timestamp(value).normalize().to_datetime64(), "D")


def build_window_features_from_df(
    user_day: pd.DataFrame,
    *,
    lookback_days: int,
    end_day: str | pd.Timestamp,
    fill_value: float = 0.0,
    dtype: str = "float32",
) -> pd.DataFrame:
    """Flatten a user-day table into fixed-length per-user features.

    Mirrors `churn_pipeline.xgb_features.build_user_window_features`, but accepts
    a DataFrame to avoid writing intermediate parquet files.

    Returns a DataFrame indexed by userId.
    """
    lookback = int(lookback_days)
    if lookback <= 0:
        raise ValueError(f"lookback_days must be > 0, got {lookback}")

    if user_day.empty:
        raise ValueError("user_day is empty")

    required = {"userId", "day"}
    missing = required - set(user_day.columns)
    if missing:
        raise ValueError(f"user_day missing required columns: {sorted(missing)}")

    user_day = user_day.sort_values(["userId", "day"], kind="mergesort")

    end_d = _to_day(end_day)
    start_d = end_d - np.timedelta64(lookback - 1, "D")

    day_d = pd.to_datetime(user_day["day"]).values.astype("datetime64[D]")
    in_window = (day_d >= start_d) & (day_d <= end_d)
    user_day = user_day.loc[in_window].copy()
    if user_day.empty:
        raise ValueError("No rows left after applying the window; check end_day/lookback_days.")

    lag = (end_d - day_d[in_window]).astype(int)
    user_day["lag"] = lag.astype(np.int16)

    feature_cols = [c for c in user_day.columns if c not in {"userId", "day", "lag"}]
    if not feature_cols:
        raise ValueError("No feature columns found (expected columns besides userId/day).")

    # Avoid exploding already-windowed/static features across all lags.
    static_cols = [c for c in feature_cols if str(c).startswith(("x_", "u_"))]
    seq_cols = [c for c in feature_cols if c not in set(static_cols)]

    pad = len(str(lookback - 1))
    parts = []

    if seq_cols:
        wide_seq = user_day.pivot(index="userId", columns="lag", values=seq_cols)
        full_cols = pd.MultiIndex.from_product([seq_cols, range(lookback)], names=["feature", "lag"])
        wide_seq = wide_seq.reindex(columns=full_cols)
        wide_seq.columns = [f"{feat}_t-{int(l):0{pad}d}" for feat, l in wide_seq.columns]
        parts.append(wide_seq)

    if static_cols:
        t0 = user_day.loc[user_day["lag"] == 0, ["userId", *static_cols]].drop_duplicates("userId", keep="last")
        wide_static = t0.set_index("userId")[static_cols]
        wide_static.columns = [f"{c}_t-{0:0{pad}d}" for c in wide_static.columns]
        parts.append(wide_static)

    wide = pd.concat(parts, axis=1)
    wide = wide.fillna(float(fill_value))
    wide = wide.sort_index()
    wide = wide.reindex(sorted(wide.columns), axis=1)
    if dtype:
        wide = wide.astype(dtype, copy=False)
    return wide


def bal_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_hat = (y_pred >= 0.5).astype(int)
    return float(balanced_accuracy_score(y_true, y_hat))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling cutoff XGB training for churn prediction")

    # Data
    p.add_argument("--competition-dir", type=str, default="churn-prediction-25-26")
    p.add_argument("--train-file", type=str, default="train.parquet")
    p.add_argument("--test-file", type=str, default="test.parquet")

    # Rolling cutoffs
    p.add_argument("--cutoff-start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--cutoff-end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--cutoff-freq", type=str, default="2D", help='pandas freq, e.g. "1D", "2D"')

    # Label definition
    p.add_argument("--horizon-days", type=int, default=10)
    p.add_argument("--label-mode", type=str, default="horizon", choices=["horizon", "ever"])

    # Feature
    p.add_argument("--lookback-days", type=int, default=51)

    # Build settings (aligned with notebooks)
    p.add_argument("--exclude-pages", type=str, default="Cancellation Confirmation,Cancel")
    p.add_argument("--exclude-auth-values", type=str, default="Cancelled")
    p.add_argument("--drop-last-k-events", type=int, default=0)
    p.add_argument("--full-calendar", action="store_true", default=True)
    p.add_argument("--val-prob-low", type=float, default=0.8)
    p.add_argument("--val-prob-high", type=float, default=1.0)
    p.add_argument("--split-seed", type=str, default="split_v1")
    p.add_argument("--cache-dir", type=str, default=".cache/user_day_features")
    p.add_argument("--output-base-dir", type=str, default="data/processed")
    p.add_argument("--batch-size", type=int, default=1_000_000)
    p.add_argument("--force-recompute", action="store_true", default=False)

    # XGB
    p.add_argument("--xgb-device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--n-estimators", type=int, default=5000)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--reg-lambda", type=float, default=2.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--early-stopping-rounds", type=int, default=100)

    # Submission
    p.add_argument(
        "--submission-cutoff",
        type=str,
        required=True,
        help="Cutoff used to build test features for submission (usually last cutoff).",
    )
    p.add_argument("--threshold", type=float, default=None, help="If omitted, uses 0.5")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    t0 = time.time()
    comp_dir = Path(args.competition_dir)
    train_path = comp_dir / args.train_file
    test_path = comp_dir / args.test_file

    cutoffs = pd.date_range(args.cutoff_start, args.cutoff_end, freq=args.cutoff_freq)
    if len(cutoffs) < 1:
        raise ValueError("No cutoffs generated; check --cutoff-start/--cutoff-end/--cutoff-freq")

    all_X_train: list[pd.DataFrame] = []
    all_y_train: list[pd.Series] = []
    all_X_val: list[pd.DataFrame] = []
    all_y_val: list[pd.Series] = []

    # 1) Build rolling train/val sets
    for cutoff in cutoffs:
        cutoff_str = str(pd.Timestamp(cutoff).date())
        paths = build_datasets_cli(
            train_path=str(train_path),
            test_path=str(test_path),
            output_dir=None,
            output_base_dir=str(args.output_base_dir),
            cache_dir=str(args.cache_dir),
            cutoff_time=cutoff_str,
            horizon_days=int(args.horizon_days),
            label_mode=str(args.label_mode),
            drop_last_k_events=int(args.drop_last_k_events),
            exclude_pages=str(args.exclude_pages),
            exclude_auth_values=str(args.exclude_auth_values),
            val_prob_low=float(args.val_prob_low),
            val_prob_high=float(args.val_prob_high),
            split_seed=str(args.split_seed),
            batch_size=int(args.batch_size),
            full_calendar=bool(args.full_calendar),
            force_recompute=bool(args.force_recompute),
            verbose=False,
        )

        train_ud = pd.read_parquet(paths["train_features"])
        val_ud = pd.read_parquet(paths["val_features"])
        train_lab = pd.read_parquet(paths["train_labels"])[["userId", "label"]]
        val_lab = pd.read_parquet(paths["val_labels"])[["userId", "label"]]

        # Window ends at cutoff-1 for features
        end_day = pd.Timestamp(cutoff).normalize() - pd.Timedelta(days=1)

        # Slice to lookback window (prevents mixing different cutoffs in flattening)
        start_day = end_day - pd.Timedelta(days=int(args.lookback_days) - 1)

        def _slice(df: pd.DataFrame) -> pd.DataFrame:
            d = pd.to_datetime(df["day"]).dt.normalize()
            return df.loc[(d >= start_day) & (d <= end_day)].copy()

        train_ud = _slice(train_ud)
        val_ud = _slice(val_ud)

        # Make sample ids unique across cutoffs
        train_ud["userId"] = train_ud["userId"].astype(str) + "|" + cutoff_str
        val_ud["userId"] = val_ud["userId"].astype(str) + "|" + cutoff_str
        train_lab["userId"] = train_lab["userId"].astype(str) + "|" + cutoff_str
        val_lab["userId"] = val_lab["userId"].astype(str) + "|" + cutoff_str

        X_tr = build_window_features_from_df(
            train_ud,
            lookback_days=int(args.lookback_days),
            end_day=end_day,
            fill_value=0.0,
            dtype="float32",
        )
        X_va = build_window_features_from_df(
            val_ud,
            lookback_days=int(args.lookback_days),
            end_day=end_day,
            fill_value=0.0,
            dtype="float32",
        )

        y_tr = train_lab.set_index("userId")["label"].reindex(X_tr.index).astype(int)
        y_va = val_lab.set_index("userId")["label"].reindex(X_va.index).astype(int)

        # Drop any unmatched rows (should be rare, but keeps it robust)
        keep_tr = y_tr.notna()
        keep_va = y_va.notna()
        X_tr = X_tr.loc[keep_tr]
        y_tr = y_tr.loc[keep_tr]
        X_va = X_va.loc[keep_va]
        y_va = y_va.loc[keep_va]

        all_X_train.append(X_tr)
        all_y_train.append(y_tr)
        all_X_val.append(X_va)
        all_y_val.append(y_va)

    X_train = pd.concat(all_X_train, axis=0)
    y_train = pd.concat(all_y_train, axis=0)
    X_val = pd.concat(all_X_val, axis=0)
    y_val = pd.concat(all_y_val, axis=0)

    # Align columns (in case some cutoff had missing feature lags)
    X_train, X_val = X_train.align(X_val, join="outer", axis=1, fill_value=0.0)

    # Scale pos weight (same heuristic as notebook)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(np.sqrt(neg / max(pos, 1)))

    xgb_params = dict(
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        min_child_weight=float(args.min_child_weight),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        reg_alpha=float(args.reg_alpha),
        gamma=float(args.gamma),
        objective="binary:logistic",
        eval_metric=bal_acc,
        tree_method="hist",
        device=str(args.xgb_device),
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    callbacks = [
        xgb.callback.EarlyStopping(
            rounds=int(args.early_stopping_rounds),
            metric_name="bal_acc",
            data_name="validation_0",
            maximize=True,
            save_best=True,
        )
    ]

    model = XGBClassifier(**xgb_params, callbacks=callbacks)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_proba))

    best_iteration = getattr(model, "best_iteration", None)
    best_n_estimators = int(model.get_params()["n_estimators"]) if best_iteration is None else int(best_iteration) + 1

    # 2) Train final model on all rolling data
    X_all = pd.concat([X_train, X_val], axis=0)
    y_all = pd.concat([y_train, y_val], axis=0)

    final_params = dict(xgb_params)
    final_params["n_estimators"] = best_n_estimators

    final_model = XGBClassifier(**final_params)
    final_model.fit(X_all, y_all, verbose=False)

    # 3) Build test features for submission (single fixed cutoff)
    sub_cutoff = pd.Timestamp(args.submission_cutoff)
    sub_cutoff_str = str(sub_cutoff.date())

    sub_paths = build_datasets_cli(
        train_path=str(train_path),
        test_path=str(test_path),
        output_dir=None,
        output_base_dir=str(args.output_base_dir),
        cache_dir=str(args.cache_dir),
        cutoff_time=sub_cutoff_str,
        horizon_days=int(args.horizon_days),
        label_mode=str(args.label_mode),
        drop_last_k_events=int(args.drop_last_k_events),
        exclude_pages=str(args.exclude_pages),
        exclude_auth_values=str(args.exclude_auth_values),
        val_prob_low=float(args.val_prob_low),
        val_prob_high=float(args.val_prob_high),
        split_seed=str(args.split_seed),
        batch_size=int(args.batch_size),
        full_calendar=bool(args.full_calendar),
        force_recompute=bool(args.force_recompute),
        verbose=False,
    )

    test_ud = pd.read_parquet(sub_paths["test_features"])
    end_day = sub_cutoff.normalize() - pd.Timedelta(days=1)

    # Slice to submission cutoff lookback window
    start_day = end_day - pd.Timedelta(days=int(args.lookback_days) - 1)
    d = pd.to_datetime(test_ud["day"]).dt.normalize()
    test_ud = test_ud.loc[(d >= start_day) & (d <= end_day)].copy()

    X_test = build_window_features_from_df(
        test_ud,
        lookback_days=int(args.lookback_days),
        end_day=end_day,
        fill_value=0.0,
        dtype="float32",
    )

    # Align test columns to training columns
    X_test = X_test.reindex(columns=X_all.columns, fill_value=0.0)

    test_proba = final_model.predict_proba(X_test)[:, 1]

    threshold = 0.5 if args.threshold is None else float(args.threshold)

    out = pd.DataFrame({"id": X_test.index.astype(int), "proba": test_proba.astype(float)})
    out["target"] = (out["proba"] >= threshold).astype(int)
    out = out[["id", "target"]].sort_values("id")

    # 4) Save artifacts
    run_name = (
        f"xgb_rolling_cut{args.cutoff_start}-{args.cutoff_end}_{args.cutoff_freq}"
        f"_sub{sub_cutoff_str}_h{args.horizon_days}_lb{args.lookback_days}"
    )

    artifact_dir = ROOT / "experiments" / "xgb_rolling" / "artifacts" / run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model.json"
    final_model.save_model(model_path)

    meta = {
        "run_name": run_name,
        "cutoffs": [str(pd.Timestamp(c).date()) for c in cutoffs],
        "submission_cutoff": sub_cutoff_str,
        "lookback_days": int(args.lookback_days),
        "horizon_days": int(args.horizon_days),
        "label_mode": str(args.label_mode),
        "scale_pos_weight": float(scale_pos_weight),
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "num_features": int(X_all.shape[1]),
        "val_auc": float(val_auc),
        "best_n_estimators": int(best_n_estimators),
        "xgb_params": xgb_params,
        "final_params": final_params,
        "threshold": float(threshold),
        "train_seconds": float(time.time() - t0),
    }
    (artifact_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    cols_path = artifact_dir / "feature_columns.json"
    cols_path.write_text(json.dumps(list(X_all.columns), ensure_ascii=False, indent=2), encoding="utf-8")

    sub_dir = ROOT / "experiments" / "xgb_rolling" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_path = sub_dir / f"{run_name}.csv"
    out.to_csv(sub_path, index=False)

    print(json.dumps({"val_auc": val_auc, "model": str(model_path), "submission": str(sub_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

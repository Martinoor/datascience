#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from churn_pipeline.dataset_builder import build_datasets_cli  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build user-day features and train/val/test splits.")
    parser.add_argument("--train-path", default="churn-prediction-25-26/train.parquet")
    parser.add_argument("--test-path", default="churn-prediction-25-26/test.parquet")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If not set, auto-generated from parameters under --output-base-dir.",
    )
    parser.add_argument("--output-base-dir", default="data/processed", help="Base dir used when --output-dir is not set.")
    parser.add_argument("--cache-dir", default=".cache/user_day_features")

    parser.add_argument(
        "--cutoff-time",
        default="",
        help="None/empty | date string like 2018-10-09 | int like 10 (per-user truncate last t days).",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=10,
        help="Label horizon length in days (only used when label-mode=horizon; when cutoff-time is int, horizon defaults to that int).",
    )
    parser.add_argument(
        "--label-mode",
        default="ever",
        choices=["horizon", "ever"],
        help="Label definition. horizon: churn in [cutoff, cutoff+horizon_days]. ever: any Cancellation Confirmation (ignores cutoff for label).",
    )
    parser.add_argument("--drop-last-k-events", type=int, default=0, help="Drop last K feature-window events per user (anti-leak).")
    parser.add_argument(
        "--exclude-pages",
        default="Cancellation Confirmation,Cancel",
        help="Comma-separated pages to exclude from features.",
    )
    parser.add_argument("--exclude-auth-values", default="Cancelled", help="Comma-separated auth values to exclude from features.")

    parser.add_argument("--val-prob-low", type=float, default=0.8)
    parser.add_argument("--val-prob-high", type=float, default=1.0)
    parser.add_argument("--split-seed", default="split_v1")

    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--no-full-calendar", action="store_true", help="Only keep days with d_events>0 (sparse user-day table).")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore cache and rebuild features.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs.")

    args = parser.parse_args()

    paths = build_datasets_cli(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        output_base_dir=args.output_base_dir,
        cache_dir=args.cache_dir,
        cutoff_time=args.cutoff_time,
        horizon_days=args.horizon_days,
        label_mode=args.label_mode,
        drop_last_k_events=args.drop_last_k_events,
        exclude_pages=args.exclude_pages,
        exclude_auth_values=args.exclude_auth_values,
        val_prob_low=args.val_prob_low,
        val_prob_high=args.val_prob_high,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        full_calendar=not args.no_full_calendar,
        force_recompute=args.force_recompute,
        verbose=not args.quiet,
    )

    print("Wrote:")
    for k, p in paths.items():
        print(f"- {k}: {p}")


if __name__ == "__main__":
    main()

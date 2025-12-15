"""Summarize Kaggle grid-search logs and write best params.

Usage:
  python scripts/summarize_kaggle_grid_best.py

It reads:
  - experiments/xgb_rolling/submission_log_xgb_rolling_grid.csv (if exists)
  - experiments/transformer_rolling/submission_log_transformer_rolling_grid.csv (if exists)

and writes:
  - experiments/best_params.json

Best is selected by highest Kaggle public `score` (numeric).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

LOGS = {
    "xgb_rolling": ROOT / "experiments" / "xgb_rolling" / "submission_log_xgb_rolling_grid.csv",
    "transformer_rolling": ROOT
    / "experiments"
    / "transformer_rolling"
    / "submission_log_transformer_rolling_grid.csv",
}

OUT_PATH = ROOT / "experiments" / "best_params.json"


def _best_row(df: pd.DataFrame) -> dict | None:
    if df.empty:
        return None
    if "score" not in df.columns:
        return None
    d = df.copy()
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d = d.dropna(subset=["score"])
    if d.empty:
        return None
    best = d.sort_values(["score"], ascending=False).iloc[0]
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in best.to_dict().items()}


def main() -> None:
    results: dict[str, dict] = {}
    for name, path in LOGS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        best = _best_row(df)
        if best is not None:
            results[name] = best

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {OUT_PATH}")
    for k, v in results.items():
        print(f"- {k}: score={v.get('score')} description={v.get('description')}")


if __name__ == "__main__":
    main()

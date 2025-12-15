from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BlendResult:
    w: float
    thr: float
    bal_acc: float


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def best_balanced_accuracy_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Return (best_thr, best_bal_acc) for decision rule y_hat = (p >= thr).

    Efficient O(N log N) via sorting.
    """
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape[0] != p.shape[0]:
        raise ValueError("y and p must be 1D arrays of same length")

    pos = int(y.sum())
    neg = int(y.shape[0] - pos)
    if pos == 0 or neg == 0:
        # Degenerate; threshold doesn't matter much. Keep simple.
        return 0.5, 0.5

    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]
    n = y_sorted.shape[0]

    # For each i in [0..n], predict positive for indices [i..n-1]
    # Compute TP(i) = sum(y[i:]) efficiently.
    tp_suffix = np.cumsum(y_sorted[::-1], dtype=np.int64)[::-1]
    tp = np.concatenate([tp_suffix, np.array([0], dtype=np.int64)])
    pred_pos = (n - np.arange(n + 1)).astype(np.int64)
    fp = pred_pos - tp
    fn = pos - tp
    tn = neg - fp

    tpr = tp / pos
    tnr = tn / neg
    bal = 0.5 * (tpr + tnr)

    best_i = int(np.argmax(bal))
    best_bal = float(bal[best_i])

    if best_i <= 0:
        best_thr = 0.0
    elif best_i >= n:
        best_thr = 1.0
    else:
        best_thr = float(p_sorted[best_i])

    return best_thr, best_bal


def grid_search_logit_blend(y: np.ndarray, p_t: np.ndarray, p_x: np.ndarray) -> BlendResult:
    z_t = _logit(p_t)
    z_x = _logit(p_x)

    def eval_w(w: float) -> BlendResult:
        z = w * z_t + (1.0 - w) * z_x
        p = _sigmoid(z)
        thr, bal = best_balanced_accuracy_threshold(y, p)
        return BlendResult(w=float(w), thr=float(thr), bal_acc=float(bal))

    # Coarse -> fine search to avoid too many sorts.
    coarse_ws = np.linspace(0.0, 1.0, 21)
    coarse = [eval_w(w) for w in coarse_ws]
    best = max(coarse, key=lambda r: r.bal_acc)

    lo = max(0.0, best.w - 0.10)
    hi = min(1.0, best.w + 0.10)
    fine_ws = np.linspace(lo, hi, 41)
    fine = [eval_w(w) for w in fine_ws]
    best = max(fine, key=lambda r: r.bal_acc)

    return best


def load_best_run_names(best_params_path: Path) -> tuple[str, str]:
    payload = json.loads(best_params_path.read_text(encoding="utf-8"))
    t_run = payload["transformer_rolling"]["run_name"]
    x_run = payload["xgb_rolling"]["run_name"]
    if not t_run or not x_run:
        raise RuntimeError("best_params.json is missing run_name for transformer_rolling or xgb_rolling")
    return str(t_run), str(x_run)


def main() -> None:
    ap = argparse.ArgumentParser(description="Blend transformer_rolling + xgb_rolling by maximizing val balanced accuracy")
    ap.add_argument("--repo", type=str, default=str(Path().resolve()), help="repo root (default: cwd)")
    ap.add_argument(
        "--best-params",
        type=str,
        default="experiments/best_params.json",
        help="path to experiments/best_params.json",
    )
    ap.add_argument("--out-dir", type=str, default="experiments/ensemble", help="output directory")
    ap.add_argument("--transformer-run", type=str, default=None, help="override transformer run_name")
    ap.add_argument("--xgb-run", type=str, default=None, help="override xgb run_name")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    best_params_path = (repo / args.best_params).resolve()
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.transformer_run and args.xgb_run:
        t_run, x_run = args.transformer_run, args.xgb_run
    else:
        t_run, x_run = load_best_run_names(best_params_path)

    t_art = repo / "experiments" / "transformer_rolling" / "artifacts" / t_run
    x_art = repo / "experiments" / "xgb_rolling" / "artifacts" / x_run

    t_val_path = t_art / "pred_val.parquet"
    x_val_path = x_art / "pred_val.parquet"
    t_test_path = t_art / "pred_test.parquet"
    x_test_path = x_art / "pred_test.parquet"

    missing = [p for p in [t_val_path, x_val_path, t_test_path, x_test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prediction files. Please run the save-artifacts cells in both notebooks to generate:\n"
            + "\n".join([str(p) for p in missing])
        )

    t_val = pd.read_parquet(t_val_path)
    x_val = pd.read_parquet(x_val_path)

    # Align by rolling userId (userId|cutoff)
    val = t_val.merge(x_val, on=["userId", "label"], how="inner", suffixes=("_t", "_x"))
    if val.empty:
        raise RuntimeError("No overlap between transformer and xgb val predictions on userId/label")

    y = val["label"].astype(int).to_numpy()
    p_t = val["proba_t"].astype(float).to_numpy()
    p_x = val["proba_x"].astype(float).to_numpy()

    best = grid_search_logit_blend(y, p_t, p_x)

    # Apply to test
    t_test = pd.read_parquet(t_test_path)
    x_test = pd.read_parquet(x_test_path)
    test = t_test.merge(x_test, on="id", how="inner", suffixes=("_t", "_x"))
    if test.empty:
        raise RuntimeError("No overlap between transformer and xgb test predictions on id")

    z = best.w * _logit(test["proba_t"].to_numpy()) + (1.0 - best.w) * _logit(test["proba_x"].to_numpy())
    p_blend = _sigmoid(z)

    sub = test[["id"]].copy()
    sub["proba"] = p_blend.astype(float)
    sub["target"] = (sub["proba"] >= float(best.thr)).astype(int)

    # Ensure full set of ids (fill missing as 0)
    example = pd.read_csv(repo / "churn-prediction-25-26" / "example_submission.csv")
    out = example[["id"]].merge(sub[["id", "target"]], on="id", how="left")
    out["target"] = out["target"].fillna(0).astype(int)

    run_tag = f"blend_logit_t={t_run}_x={x_run}_w{best.w:.3f}_thr{best.thr:.3f}_bal{best.bal_acc:.6f}"
    sub_dir = out_dir / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    out_path = sub_dir / f"{run_tag}.csv"
    out.to_csv(out_path, index=False)

    meta = {
        "transformer_run": t_run,
        "xgb_run": x_run,
        "val_rows_overlap": int(val.shape[0]),
        "best_w": float(best.w),
        "best_thr": float(best.thr),
        "best_val_bal_acc": float(best.bal_acc),
        "method": "logit_blend_gridsearch_bal_acc",
    }
    (out_dir / f"{run_tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("val overlap rows:", val.shape[0])
    print("best w:", best.w, "best thr:", best.thr, "best val bal_acc:", best.bal_acc)
    print("submission:", out_path)


if __name__ == "__main__":
    main()

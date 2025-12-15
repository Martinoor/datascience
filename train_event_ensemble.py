from __future__ import annotations

"""
Train the existing event-sequence Transformer with seed ensembling (average voting).

Outputs:
- validation balanced accuracy (and AUC for reference)
- best threshold by balanced accuracy
- `submission_event_ensemble.csv` (id,target with 0/1)
- run artifacts under `runs/event_ensemble/<run_id>/` (logs, configs, per-seed plots/metrics)

Run:
  python train_event_ensemble.py
"""
#%%
import json
import io
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from kaggle_submit import submit_and_track
from feature_pipeline import prepare_datasets
from transformer_model import (
    ChurnTransformer,
    TrainingConfig,
    UserSequenceDataset,
    average_prob_dicts,
    build_user_sequences,
    collate_batch,
    make_dataloaders,
    predict_proba,
    roc_auc_from_probs,
    set_seed,
    train_model,
)

#%%
class _Tee(io.TextIOBase):
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, s: str) -> int:  # type: ignore[override]
        n = 0
        for stream in self._streams:
            try:
                n = stream.write(s)
            except Exception:
                # Best-effort: don't crash training if a stream fails.
                pass
        return n

    def flush(self) -> None:  # type: ignore[override]
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self) -> bool:  # type: ignore[override]
        for stream in self._streams:
            try:
                return bool(stream.isatty())
            except Exception:
                continue
        return False


@contextmanager
def _tee_stdio(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, f)
        sys.stderr = _Tee(old_err, f)
        try:
            yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout, sys.stderr = old_out, old_err


def _utc_now_str() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _try_git_head() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prob_dict_to_frame(probs: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"userId": list(probs.keys()), "prob": list(probs.values())})


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int32, copy=False)
    y_pred = y_pred.astype(np.int32, copy=False)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(0.5 * (tpr + tnr))


def _best_threshold_by_balanced_accuracy(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99, dtype=np.float32)
    best_thr = 0.5
    best_bacc = -1.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.int32)
        bacc = _balanced_accuracy(y_true, y_pred)
        if bacc > best_bacc:
            best_bacc = bacc
            best_thr = float(thr)
    return best_thr, float(best_bacc)


def _to_submission_frame(probs: dict[str, float], threshold: float) -> pd.DataFrame:
    ids = list(probs.keys())
    p = np.array([probs[i] for i in ids], dtype=np.float32)
    target = (p >= threshold).astype(int)
    try:
        ids_out = pd.to_numeric(pd.Series(ids), errors="raise").astype(int)
    except Exception:
        ids_out = pd.Series(ids)
    out = pd.DataFrame({"id": ids_out, "target": target})
    return out.sort_values("id").reset_index(drop=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = "churn-prediction-25-26/train.parquet"
    test_path = "churn-prediction-25-26/test.parquet"
    competition = "churn-prediction-25-26"
    submit_to_kaggle = True

    dataset_params = {
        "val_ratio": 0.2,
        "random_state": 42,
        "truncate_buffer_min": 2,
        "truncate_buffer_frac": 0.02,
        "cutoff_time": "2018-11-09",
        "use_cache": True,
    }
    config = TrainingConfig(
        d_model=256,
        nhead=4,
        num_layers=4,            
        dim_feedforward=768,    
        dropout=0.1,           

        max_seq_len=450,       
        batch_size=256,
        num_workers=4,

        lr=1e-5,                 
        weight_decay=1.5e-3,     
        epochs=60,

        use_cosine_decay=True,
        eta_min_factor=0.05,
        warmup_epochs=3,
        use_focal_loss=True,
        focal_gamma=1.6,
        
        early_stop_patience=4,
        early_stop_min_delta=1e-4,
        pooling="attn",
        monitor = 'val_auc'
    )

    # pos_weight = float((labels == 0).sum() / max(1, (labels == 1).sum()))
    pos_weight = 2.0
    seeds = [42, 43, 44]

    run_started_utc = _utc_now_str()
    run_id = (
        f"{run_started_utc.replace(':', '').replace('-', '').replace('T', '_')}"
        f"_ens{len(seeds)}_pool={config.pooling}"
        f"_dm={config.d_model}_l={config.num_layers}_ff={config.dim_feedforward}"
        f"_do={config.dropout}_lr={config.lr:g}_wd={config.weight_decay:g}"
        f"_seq={config.max_seq_len}"
    )
    run_dir = Path("runs") / "event_ensemble" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    meta_path = run_dir / "run_meta.json"
    base_meta: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "run_started_utc": run_started_utc,
        "script": Path(__file__).name,
        "argv": list(sys.argv),
        "git_head": _try_git_head(),
        "device": str(device),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", ""),
        "numpy": getattr(np, "__version__", ""),
        "pandas": getattr(pd, "__version__", ""),
        "train_path": train_path,
        "test_path": test_path,
        "dataset_params": dataset_params,
        "training_config": asdict(config),
        "pos_weight": float(pos_weight),
        "seeds": list(seeds),
        "submit_to_kaggle": bool(submit_to_kaggle),
        "competition": competition,
    }
    _write_json(meta_path, base_meta)

    with _tee_stdio(run_dir / "console.log"):
        print(f"run_dir: {run_dir}")
        print(f"run_started_utc: {run_started_utc}")
        print("dataset_params:", dataset_params)
        print("training_config:", asdict(config))
        print("seeds:", seeds)

        train_df, val_df, test_df, labels, artifacts = prepare_datasets(
            train_path,
            test_path,
            val_ratio=dataset_params["val_ratio"],
            random_state=dataset_params["random_state"],
            truncate_buffer_min=dataset_params["truncate_buffer_min"],
            truncate_buffer_frac=dataset_params["truncate_buffer_frac"],
            cutoff_time=dataset_params["cutoff_time"],
            use_cache=dataset_params["use_cache"],
        )

        train_sequences = build_user_sequences(train_df, artifacts.numeric_cols, config.max_seq_len)
        val_sequences = build_user_sequences(val_df, artifacts.numeric_cols, config.max_seq_len)
        test_sequences = build_user_sequences(test_df, artifacts.numeric_cols, config.max_seq_len)

        train_loader, val_loader = make_dataloaders(
            train_sequences, val_sequences, labels.to_dict(), config
        )
        test_loader = torch.utils.data.DataLoader(
            UserSequenceDataset(test_sequences, None, list(test_sequences.keys())),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_batch,
        )

        labels_dict = labels.to_dict()
        seed_records: list[dict[str, Any]] = []
        val_prob_dicts: list[dict[str, float]] = []
        test_prob_dicts: list[dict[str, float]] = []

        # Save dataset + feature info for tuning/debugging.
        base_meta.update(
            {
                "n_train_events": int(train_df.shape[0]),
                "n_val_events": int(val_df.shape[0]),
                "n_test_events": int(test_df.shape[0]),
                "n_train_users": int(len(train_sequences)),
                "n_val_users": int(len(val_sequences)),
                "n_test_users": int(len(test_sequences)),
                "numeric_cols": list(artifacts.numeric_cols),
            }
        )
        _write_json(meta_path, base_meta)

        for seed in seeds:
            seed_dir = run_dir / f"seed={seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            set_seed(seed)
            model = ChurnTransformer(
                num_numeric=len(artifacts.numeric_cols),
                num_pages=len(artifacts.page_categories),
                num_metro=len(artifacts.metro_mapping),
                num_state=len(artifacts.state_mapping),
                num_device=len(artifacts.device_mapping),
                config=config,
            ).to(device)

            t0 = time.time()
            train_losses, val_losses, model, best_epoch, best_state = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=config.epochs,
                lr=config.lr,
                weight_decay=config.weight_decay,
                pos_weight=pos_weight,
                use_cosine_decay=config.use_cosine_decay,
                eta_min_factor=config.eta_min_factor,
                warmup_epochs=config.warmup_epochs,
                use_focal_loss=config.use_focal_loss,
                focal_gamma=config.focal_gamma,
                early_stop_patience=config.early_stop_patience,
                early_stop_min_delta=config.early_stop_min_delta,
                monitor=config.monitor,
            )
            seed_seconds = float(time.time() - t0)

            best_state_cpu = {k: v.detach().cpu() for k, v in best_state.items()}
            model_path = seed_dir / "model_state.pt"
            torch.save(best_state_cpu, model_path)

            print(f"[seed={seed}] best_epoch={best_epoch} seconds={seed_seconds:.1f}")
            val_probs = predict_proba(model, val_loader, device)
            test_probs = predict_proba(model, test_loader, device)

            val_prob_dicts.append(val_probs)
            test_prob_dicts.append(test_probs)

            # Persist predictions for later analysis/tuning.
            _prob_dict_to_frame(val_probs).to_parquet(seed_dir / "val_probs.parquet", index=False)
            _prob_dict_to_frame(test_probs).to_parquet(seed_dir / "test_probs.parquet", index=False)

            # Seed-level metrics.
            seed_auc = roc_auc_from_probs(labels_dict, val_probs)
            val_users_seed = list(val_probs.keys())
            y_true_seed = np.array([int(labels_dict[uid]) for uid in val_users_seed], dtype=np.int32)
            y_prob_seed = np.array([val_probs[uid] for uid in val_users_seed], dtype=np.float32)
            seed_thr, seed_bacc = _best_threshold_by_balanced_accuracy(y_true_seed, y_prob_seed)

            loss_df = pd.DataFrame(
                {
                    "epoch": np.arange(len(train_losses), dtype=np.int32),
                    "train_loss": train_losses,
                    "val_loss": val_losses,
                }
            )
            loss_df.to_csv(seed_dir / "loss_curve.csv", index=False)

            plt.figure(figsize=(6, 4))
            plt.plot(train_losses, label="train")
            plt.plot(val_losses, label="val")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(seed_dir / "loss_curve.png", dpi=150)
            plt.close()

            seed_record = {
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "seconds": seed_seconds,
                "val_auc": float(seed_auc),
                "val_balanced_accuracy": float(seed_bacc),
                "best_threshold": float(seed_thr),
                "train_loss_last": float(train_losses[-1]) if train_losses else float("nan"),
                "val_loss_last": float(val_losses[-1]) if val_losses else float("nan"),
                "val_loss_min": float(min(val_losses)) if val_losses else float("nan"),
                "model_state_path": str(model_path),
                "val_probs_path": str(seed_dir / "val_probs.parquet"),
                "test_probs_path": str(seed_dir / "test_probs.parquet"),
                "loss_curve_png": str(seed_dir / "loss_curve.png"),
                "loss_curve_csv": str(seed_dir / "loss_curve.csv"),
            }
            seed_records.append(seed_record)
            _write_json(seed_dir / "seed_meta.json", seed_record)

        pd.DataFrame(seed_records).to_csv(run_dir / "seed_metrics.csv", index=False)

        val_probs_avg = average_prob_dicts(val_prob_dicts)
        test_probs_avg = average_prob_dicts(test_prob_dicts)

        _prob_dict_to_frame(val_probs_avg).to_parquet(run_dir / "val_probs_avg.parquet", index=False)
        _prob_dict_to_frame(test_probs_avg).to_parquet(run_dir / "test_probs_avg.parquet", index=False)

        auc = roc_auc_from_probs(labels_dict, val_probs_avg)
        val_users = list(val_probs_avg.keys())
        y_true = np.array([int(labels_dict[uid]) for uid in val_users], dtype=np.int32)
        y_prob = np.array([val_probs_avg[uid] for uid in val_users], dtype=np.float32)
        best_thr, best_bacc = _best_threshold_by_balanced_accuracy(y_true, y_prob)

        print(
            f"val_balanced_accuracy={best_bacc:.6f} best_thr={best_thr:.3f} val_auc={auc:.6f}"
        )
        print(
            "config:",
            {
                k: v
                for k, v in asdict(config).items()
                if k
                in {
                    "d_model",
                    "nhead",
                    "num_layers",
                    "dim_feedforward",
                    "dropout",
                    "max_seq_len",
                    "batch_size",
                    "lr",
                    "weight_decay",
                    "epochs",
                    "pooling",
                }
            },
        )

        sub = _to_submission_frame(test_probs_avg, best_thr)
        out_path_run = run_dir / "submission_event_ensemble.csv"
        out_path_root = Path("submission_event_ensemble.csv")
        sub.to_csv(out_path_run, index=False)
        sub.to_csv(out_path_root, index=False)
        print(f"wrote: {out_path_run}")
        print(f"wrote: {out_path_root}")

        run_note = (
            f"event_ens{len(seeds)}_pool={config.pooling}_thr={best_thr:.3f}"
            f"_bacc={best_bacc:.4f}_auc={auc:.4f}_do={config.dropout}"
            f"_lr={config.lr:g}_wd={config.weight_decay:g}"
        )

        final_summary: dict[str, Any] = {
            "run_finished_utc": _utc_now_str(),
            "run_note": run_note,
            "val_auc": float(auc),
            "val_balanced_accuracy": float(best_bacc),
            "best_threshold": float(best_thr),
        }
        _write_json(run_dir / "ensemble_metrics.json", final_summary)

        submit_info: dict[str, Any] | None = None
        if submit_to_kaggle:
            submit_info = submit_and_track(
                out_path_run,
                competition,
                run_note,
                extra_meta={
                    "model": "event_transformer_ensemble",
                    "seeds": ",".join(map(str, seeds)),
                    "pooling": config.pooling,
                    "best_threshold": float(best_thr),
                    "val_auc": float(auc),
                    "val_balanced_accuracy": float(best_bacc),
                    "max_seq_len": int(config.max_seq_len),
                    "d_model": int(config.d_model),
                    "nhead": int(config.nhead),
                    "num_layers": int(config.num_layers),
                    "dim_feedforward": int(config.dim_feedforward),
                    "dropout": float(config.dropout),
                    "lr": float(config.lr),
                    "weight_decay": float(config.weight_decay),
                    "epochs": int(config.epochs),
                },
            )
            _write_json(run_dir / "kaggle_submission.json", submit_info)

        # Append a compact run index for quick tuning iteration.
        index_path = run_dir.parent / "runs_index.csv"
        summary_row = {
            "run_started_utc": run_started_utc,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "run_note": run_note,
            "seeds": ",".join(map(str, seeds)),
            "val_auc": float(auc),
            "val_balanced_accuracy": float(best_bacc),
            "best_threshold": float(best_thr),
            "kaggle_status": None if not submit_info else submit_info.get("status"),
            "kaggle_score": None if not submit_info else submit_info.get("score"),
            "d_model": int(config.d_model),
            "nhead": int(config.nhead),
            "num_layers": int(config.num_layers),
            "dim_feedforward": int(config.dim_feedforward),
            "dropout": float(config.dropout),
            "max_seq_len": int(config.max_seq_len),
            "batch_size": int(config.batch_size),
            "lr": float(config.lr),
            "weight_decay": float(config.weight_decay),
            "epochs": int(config.epochs),
            "pooling": str(config.pooling),
            "pos_weight": float(pos_weight),
            **dataset_params,
        }
        new_df = pd.DataFrame([summary_row])
        if index_path.exists():
            old_df = pd.read_csv(index_path)
            out_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            out_df = new_df
        out_df.to_csv(index_path, index=False)

        base_meta.update(final_summary)
        if submit_info is not None:
            base_meta["kaggle_submission"] = submit_info
        base_meta["seed_metrics_path"] = str(run_dir / "seed_metrics.csv")
        _write_json(meta_path, base_meta)
    

#%%
if __name__ == "__main__":
    main()

# %%

from __future__ import annotations

"""
Train the existing event-sequence Transformer with seed ensembling (average voting).

Outputs:
- validation AUC
- best threshold by F1
- `submission_event_ensemble.csv` (id,target with 0/1)

Run:
  python train_event_ensemble.py
"""
#%%
from dataclasses import asdict
from pathlib import Path

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
def _best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91, dtype=np.float32)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.int32)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)


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

    train_df, val_df, test_df, labels, artifacts = prepare_datasets(
            train_path,
            test_path,
            val_ratio=0.2,
            random_state=42,
            truncate_buffer_min=2,
            truncate_buffer_frac=0.02,
            cutoff_time='2018-11-09',
            use_cache = True
        )

    config = TrainingConfig(
        d_model=256,
        nhead=4,
        num_layers=4,            
        dim_feedforward=768,    
        dropout=0.1,           

        max_seq_len=450,       
        batch_size=256,
        num_workers=4,

        lr=6e-4,                 
        weight_decay=1.5e-3,     
        epochs=35,

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

    train_sequences = build_user_sequences(train_df, artifacts.numeric_cols, config.max_seq_len)
    val_sequences = build_user_sequences(val_df, artifacts.numeric_cols, config.max_seq_len)
    test_sequences = build_user_sequences(test_df, artifacts.numeric_cols, config.max_seq_len)

    train_loader, val_loader = make_dataloaders(train_sequences, val_sequences, labels.to_dict(), config)
    test_loader = torch.utils.data.DataLoader(
        UserSequenceDataset(test_sequences, None, list(test_sequences.keys())),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
    )

    seeds = [42, 43, 44]
    val_prob_dicts: list[dict[str, float]] = []
    test_prob_dicts: list[dict[str, float]] = []

    for seed in seeds:
        set_seed(seed)
        model = ChurnTransformer(
            num_numeric=len(artifacts.numeric_cols),
            num_pages=len(artifacts.page_categories),
            num_metro=len(artifacts.metro_mapping),
            num_state=len(artifacts.state_mapping),
            num_device=len(artifacts.device_mapping),
            config=config,
        ).to(device)

        train_losses, val_losses, model, best_epoch, _ = train_model(
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
            monitor = config.monitor
        )

        print(f"[seed={seed}] best_epoch={best_epoch}")
        val_probs = predict_proba(model, val_loader, device)
        test_probs = predict_proba(model, test_loader, device)
        val_prob_dicts.append(val_probs)
        test_prob_dicts.append(test_probs)
        plt.figure(figsize=(6,4))
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_loss.png', dpi=150)
        plt.show()

    val_probs_avg = average_prob_dicts(val_prob_dicts)
    test_probs_avg = average_prob_dicts(test_prob_dicts)

    auc = roc_auc_from_probs(labels.to_dict(), val_probs_avg)
    val_users = list(val_probs_avg.keys())
    y_true = np.array([int(labels[uid]) for uid in val_users], dtype=np.int32)
    y_prob = np.array([val_probs_avg[uid] for uid in val_users], dtype=np.float32)
    best_thr, best_f1 = _best_threshold_by_f1(y_true, y_prob)

    print(f"val_auc={auc:.6f} best_thr={best_thr:.3f} best_f1={best_f1:.6f}")
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
    out_path = Path("submission_event_ensemble.csv")
    sub.to_csv(out_path, index=False)
    print(f"wrote: {out_path}")

    if submit_to_kaggle:
        run_note = (
            f"event_ens{len(seeds)}_pool={config.pooling}_thr={best_thr:.3f}"
            f"_auc={auc:.4f}_do={config.dropout}_lr={config.lr:g}_wd={config.weight_decay:g}"
        )
        submit_and_track(
            out_path,
            competition,
            run_note,
            extra_meta={
                "model": "event_transformer_ensemble",
                "seeds": ",".join(map(str, seeds)),
                "pooling": config.pooling,
                "best_threshold": float(best_thr),
                "val_auc": float(auc),
                "val_best_f1": float(best_f1),
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
    

#%%
if __name__ == "__main__":
    main()

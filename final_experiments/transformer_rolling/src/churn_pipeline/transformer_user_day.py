"""
Transformer model utilities for churn prediction on user-day sequences.

This mirrors the logic in `/Data/yuhan.wu/py_kaggle/transformer_model.py` but
adapts the input to the daily aggregated `user-day` table produced by this
project (numeric features only).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _balanced_accuracy_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy for binary labels/preds.

    y_true: array of {0,1}
    y_pred: array of {0,1}
    """
    y_true = y_true.astype(np.int8, copy=False)
    y_pred = y_pred.astype(np.int8, copy=False)
    pos = y_true == 1
    neg = ~pos

    tp = int(np.sum((y_pred == 1) & pos))
    fn = int(np.sum((y_pred == 0) & pos))
    tn = int(np.sum((y_pred == 0) & neg))
    fp = int(np.sum((y_pred == 1) & neg))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr)


def _best_balanced_accuracy_from_logits(
    probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    """Compute best balanced accuracy over a threshold grid.

    This is written to avoid storing a huge (N, K) matrix by accumulating counts
    per threshold in a streaming fashion.
    """
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim != 1 or thresholds.size == 0:
        # fallback to default 0.5
        thr = 0.5
        bal = _balanced_accuracy_binary(y_true, (probs >= thr).astype(np.int8))
        return float(bal), float(thr)

    tp = np.zeros((thresholds.size,), dtype=np.int64)
    tn = np.zeros((thresholds.size,), dtype=np.int64)
    fp = np.zeros((thresholds.size,), dtype=np.int64)
    fn = np.zeros((thresholds.size,), dtype=np.int64)

    # batch-streamed update
    # (caller typically passes per-batch probs/y; this helper supports whole arrays too)
    probs = np.asarray(probs, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int8)

    # If whole arrays: do one vectorized pass
    # pred shape: (N, K)
    pred = probs[:, None] >= thresholds[None, :]
    pos = y_true == 1
    neg = ~pos

    tp += np.sum(pred & pos[:, None], axis=0, dtype=np.int64)
    fp += np.sum(pred & neg[:, None], axis=0, dtype=np.int64)
    fn += np.sum((~pred) & pos[:, None], axis=0, dtype=np.int64)
    tn += np.sum((~pred) & neg[:, None], axis=0, dtype=np.int64)

    tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) > 0)
    tnr = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=np.float64), where=(tn + fp) > 0)
    bal = 0.5 * (tpr + tnr)

    best_idx = int(np.argmax(bal))
    return float(bal[best_idx]), float(thresholds[best_idx])


@dataclass
class TrainingConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.2

    max_seq_len: int = 51
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 12
    num_workers: int = 0

    use_cosine_decay: bool = False
    eta_min_factor: float = 0.1
    warmup_epochs: int = 0

    use_focal_loss: bool = False
    focal_gamma: float = 2.0

    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0

    # Pooling over time dimension for sequence representation.
    # - "mean": masked mean pooling (default, backward compatible)
    # - "last": last valid (non-pad) timestep
    # - "recency": exponentially recency-weighted pooling
    pooling: str = "mean"
    recency_gamma: float = 0.15


@dataclass
class SequenceData:
    numeric: np.ndarray


class UserSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: Dict[str, SequenceData],
        labels: Optional[Dict[str, int]],
        user_ids: List[str],
    ):
        self.sequences = sequences
        self.labels = labels
        self.user_ids = user_ids

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int):
        user_id = self.user_ids[idx]
        seq = self.sequences[user_id]
        label = -1.0 if self.labels is None else float(self.labels[user_id])
        return (
            seq.numeric.astype(np.float32),
            label,
            user_id,
        )


class UserDayTableSequenceDataset(Dataset):
    """Memory-lean Dataset that builds sequences on-the-fly from a user-day table.

    Why this exists:
    - `build_user_day_sequences()` materializes a `dict[user_id -> np.ndarray]`, which
      is very memory-heavy for large rolling datasets (Python dict + many small arrays).
    - This dataset stores only:
        - per-user (start, end) row ranges
        - column views for each feature (no 2D materialization)
      and constructs the (<= max_seq_len) sequence only when a sample is fetched.
    """

    def __init__(
        self,
        df: "pd.DataFrame",
        feature_cols: List[str],
        max_seq_len: int,
        labels: Optional[Dict[str, int]],
        *,
        assume_sorted: bool = False,
    ):
        self.feature_cols = list(feature_cols)
        self.max_seq_len = int(max_seq_len)
        self.labels = labels

        if not assume_sorted:
            # Stable sort keeps deterministic ordering for equal keys.
            df = df.sort_values(["userId", "day"], kind="mergesort")
        df = df.reset_index(drop=True)
        self._df = df

        uid_rows = df["userId"].to_numpy(copy=False)
        n = int(len(uid_rows))
        if n == 0:
            self.user_ids: List[str] = []
            self._starts = np.zeros((0,), dtype=np.int64)
            self._ends = np.zeros((0,), dtype=np.int64)
            self._cols = []
            return

        # Find contiguous groups in the sorted table.
        # uid_rows is object dtype; comparisons are fine and memory-light.
        changes = uid_rows[1:] != uid_rows[:-1]
        starts = np.concatenate([np.array([0], dtype=np.int64), np.nonzero(changes)[0].astype(np.int64) + 1])
        ends = np.concatenate([starts[1:], np.array([n], dtype=np.int64)])

        # Store user ids once per group (not per row).
        self.user_ids = [str(uid_rows[s]) for s in starts.tolist()]
        self._starts = starts
        self._ends = ends

        # Keep per-column views (no 2D dense matrix allocation).
        self._cols = [df[c].to_numpy(copy=False) for c in self.feature_cols]

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int):
        uid = self.user_ids[idx]
        start = int(self._starts[idx])
        end = int(self._ends[idx])

        # Truncate to most recent max_seq_len rows.
        length = end - start
        take = min(length, self.max_seq_len)
        s = end - take

        x = np.empty((take, len(self._cols)), dtype=np.float32)
        for j, col in enumerate(self._cols):
            # Assignment casts to float32 if needed, without large temporaries.
            x[:, j] = col[s:end]

        label = -1.0 if self.labels is None else float(self.labels[uid])
        return x, label, uid


def build_user_day_sequences(
    df: "pd.DataFrame",
    feature_cols: List[str],
    max_seq_len: int,
) -> Dict[str, SequenceData]:
    """
    Convert a user-day dataframe into per-user numeric sequences truncated to the
    most recent `max_seq_len` days.
    """
    sequences: Dict[str, SequenceData] = {}
    grouped = df.sort_values(["userId", "day"]).groupby("userId", sort=False)
    for user_id, g in tqdm(grouped, total=grouped.ngroups, desc="build_user_day_sequences"):
        g = g.tail(max_seq_len)
        sequences[str(user_id)] = SequenceData(
            numeric=g[feature_cols].to_numpy(np.float32),
        )
    return sequences


def filter_nonconstant_feature_cols(
    df: "pd.DataFrame",
    feature_cols: List[str],
    *,
    min_std: float = 1e-8,
) -> List[str]:
    """Filter out feature columns with ~zero variance on the provided dataframe.

    This is especially important for Transformer inputs when vocab is built from
    train∪test and some categories only appear in test (train column is all zeros).
    """
    if not feature_cols:
        return []
    x = df[feature_cols].to_numpy(np.float32, copy=False)
    std = np.nanstd(x, axis=0)
    keep = std > float(min_std)
    return [c for c, k in zip(feature_cols, keep.tolist()) if k]


def collate_batch(batch):
    numeric_seq, labels, user_ids = zip(*batch)
    batch_size = len(batch)
    seq_lens = [len(x) for x in numeric_seq]
    max_len = max(seq_lens)
    num_feat_dim = numeric_seq[0].shape[1]

    num_tensor = torch.zeros(batch_size, max_len, num_feat_dim, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    for i, n in enumerate(numeric_seq):
        length = len(n)
        num_tensor[i, :length] = torch.from_numpy(n)
        padding_mask[i, :length] = False  # False = keep, True = pad

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return (
        num_tensor,
        padding_mask,
        labels_tensor,
        list(user_ids),
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class ChurnTransformerUserDay(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        config: TrainingConfig,
    ):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(num_numeric, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=config.max_seq_len + 50)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, numeric_feats: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(numeric_feats)
        x = self.input_dropout(x)
        x = self.pos_encoder(x)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        valid = (~padding_mask).float()  # (B,T)
        pooling = str(getattr(self.config, "pooling", "mean"))

        if pooling == "last":
            lengths = valid.sum(dim=1).long().clamp(min=1)  # (B,)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, encoded.size(-1))
            pooled = encoded.gather(dim=1, index=idx).squeeze(1)
        elif pooling == "recency":
            gamma = float(getattr(self.config, "recency_gamma", 0.15))
            t = torch.arange(encoded.size(1), device=encoded.device, dtype=encoded.dtype)  # (T,)
            w = torch.exp(gamma * (t - t.max())).view(1, -1)  # (1,T)
            wv = w * valid
            denom = wv.sum(dim=1, keepdim=True).clamp(min=1e-6)
            pooled = (encoded * wv.unsqueeze(-1)).sum(dim=1) / denom
        else:
            # masked mean pooling
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (encoded * valid.unsqueeze(-1)).sum(dim=1) / denom

        logits = self.classifier(pooled).squeeze(-1)
        return logits


def make_dataloaders(
    train_sequences: Dict[str, SequenceData],
    val_sequences: Dict[str, SequenceData],
    labels: Dict[str, int],
    config: TrainingConfig,
    *,
    train_generator: torch.Generator | None = None,
):
    train_dataset = UserSequenceDataset(train_sequences, labels, list(train_sequences.keys()))
    val_dataset = UserSequenceDataset(val_sequences, labels, list(val_sequences.keys()))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader


def make_dataloaders_from_user_day_tables(
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
    feature_cols: List[str],
    labels: Dict[str, int],
    config: TrainingConfig,
    *,
    max_seq_len: int | None = None,
    train_generator: torch.Generator | None = None,
    assume_sorted: bool = False,
):
    """Create DataLoaders without materializing per-user sequence dicts."""
    msl = int(config.max_seq_len if max_seq_len is None else max_seq_len)
    train_dataset = UserDayTableSequenceDataset(
        train_df,
        feature_cols,
        msl,
        labels,
        assume_sorted=assume_sorted,
    )
    val_dataset = UserDayTableSequenceDataset(
        val_df,
        feature_cols,
        msl,
        labels,
        assume_sorted=assume_sorted,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader, train_dataset.user_ids, val_dataset.user_ids


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    pos_weight: float,
    use_cosine_decay: bool = False,
    eta_min_factor: float = 0.1,
    warmup_epochs: int = 0,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    *,
    select_best: str = "loss",
    early_stop_metric: str | None = None,
    bal_acc_thresholds: np.ndarray | None = None,
) -> Tuple[List[float], List[float], nn.Module, int, Dict[str, torch.Tensor]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=lr * eta_min_factor)
        if use_cosine_decay
        else None
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")
    select_best = str(select_best).lower().strip()
    stop_metric = select_best if early_stop_metric is None else str(early_stop_metric).lower().strip()
    allowed = {"loss", "bal_acc"}
    if select_best not in allowed:
        raise ValueError(f"select_best must be one of {sorted(allowed)}, got {select_best!r}")
    if stop_metric not in allowed:
        raise ValueError(f"early_stop_metric must be one of {sorted(allowed)}, got {stop_metric!r}")

    if bal_acc_thresholds is None:
        bal_acc_thresholds = np.linspace(0.01, 0.99, 99, dtype=np.float32)

    # Best checkpoint selection
    best_select = float("inf") if select_best == "loss" else float("-inf")
    # Early-stop tracking (can be different from select_best)
    best_stop = float("inf") if stop_metric == "loss" else float("-inf")

    best_state = deepcopy(model.state_dict())
    best_epoch = -1
    train_history: List[float] = []
    val_history: List[float] = []
    no_improve = 0

    for epoch in tqdm(range(epochs), desc="train_model"):
        model.train()
        running = 0.0
        for batch in train_loader:
            (num_feats, padding_mask, labels, _) = batch
            num_feats = num_feats.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(num_feats, padding_mask)
            if use_focal_loss:
                probs = torch.sigmoid(logits)
                pt = probs * labels + (1 - probs) * (1 - labels)
                loss = bce(logits, labels) * torch.pow(1 - pt, focal_gamma)
                loss = loss.mean()
            else:
                loss = bce(logits, labels).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * labels.size(0)
        train_loss = running / len(train_loader.dataset)
        train_history.append(train_loss)

        model.eval()
        val_running = 0.0

        need_bal = (select_best == "bal_acc") or (stop_metric == "bal_acc")
        if need_bal:
            thr = np.asarray(bal_acc_thresholds, dtype=np.float32)
            tp = np.zeros((thr.size,), dtype=np.int64)
            tn = np.zeros((thr.size,), dtype=np.int64)
            fp = np.zeros((thr.size,), dtype=np.int64)
            fn = np.zeros((thr.size,), dtype=np.int64)

        with torch.no_grad():
            for batch in val_loader:
                (num_feats, padding_mask, labels, _) = batch
                num_feats = num_feats.to(device)
                padding_mask = padding_mask.to(device)
                labels = labels.to(device)

                logits = model(num_feats, padding_mask)
                if use_focal_loss:
                    probs = torch.sigmoid(logits)
                    pt = probs * labels + (1 - probs) * (1 - labels)
                    loss = bce(logits, labels) * torch.pow(1 - pt, focal_gamma)
                    loss = loss.mean()
                else:
                    loss = bce(logits, labels).mean()
                val_running += loss.item() * labels.size(0)

                if need_bal:
                    # Update per-threshold confusion counts without storing all probs.
                    probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False)
                    y = labels.detach().cpu().numpy().astype(np.int8, copy=False)
                    pred = probs[:, None] >= thr[None, :]
                    pos = y == 1
                    neg = ~pos
                    tp += np.sum(pred & pos[:, None], axis=0, dtype=np.int64)
                    fp += np.sum(pred & neg[:, None], axis=0, dtype=np.int64)
                    fn += np.sum((~pred) & pos[:, None], axis=0, dtype=np.int64)
                    tn += np.sum((~pred) & neg[:, None], axis=0, dtype=np.int64)

        val_loss = val_running / len(val_loader.dataset)
        val_history.append(val_loss)

        # Derive val_bal_acc if requested
        if need_bal:
            tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) > 0)
            tnr = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=np.float64), where=(tn + fp) > 0)
            bal = 0.5 * (tpr + tnr)
            best_bal_acc = float(bal[int(np.argmax(bal))]) if bal.size else 0.0
        else:
            best_bal_acc = 0.0

        # Update best checkpoint (selection metric)
        select_score = val_loss if select_best == "loss" else best_bal_acc
        if (select_best == "loss" and select_score < best_select - early_stop_min_delta) or (
            select_best == "bal_acc" and select_score > best_select + early_stop_min_delta
        ):
            best_select = float(select_score)
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch

        # Early stopping tracking (stop metric)
        stop_score = val_loss if stop_metric == "loss" else best_bal_acc
        improved_stop = (stop_metric == "loss" and stop_score < best_stop - early_stop_min_delta) or (
            stop_metric == "bal_acc" and stop_score > best_stop + early_stop_min_delta
        )
        if improved_stop:
            best_stop = float(stop_score)
            no_improve = 0
        else:
            no_improve += 1

        if warmup_epochs > 0 and epoch < warmup_epochs:
            new_lr = lr * float(epoch + 1) / float(warmup_epochs)
            for g in optimizer.param_groups:
                g["lr"] = new_lr
        elif scheduler is not None:
            scheduler.step()

        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            break

    model.load_state_dict(best_state)
    return train_history, val_history, model, best_epoch, best_state


def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    probs: Dict[str, float] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict_proba"):
            (num_feats, padding_mask, _, user_ids) = batch
            num_feats = num_feats.to(device)
            padding_mask = padding_mask.to(device)
            logits = model(num_feats, padding_mask)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            for uid, prob in zip(user_ids, batch_probs):
                probs[str(uid)] = float(prob)
    return probs

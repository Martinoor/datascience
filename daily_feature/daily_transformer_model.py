from __future__ import annotations

"""
Transformer utilities for daily (user × day) sequences.

The daily token = one day, with:
- numeric daily aggregates (listen counts, session stats, etc.)
- categorical IDs repeated per day (device/metro/state) + day-of-week ID
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from transformer_model import PositionalEncoding, TrainingConfig


@dataclass
class DailySequenceData:
    numeric: np.ndarray
    dow_ids: np.ndarray
    metro_ids: np.ndarray
    state_ids: np.ndarray
    device_ids: np.ndarray


class UserDaySequenceDataset(Dataset):
    def __init__(
        self,
        sequences: Dict[str, DailySequenceData],
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
            seq.dow_ids.astype(np.int64),
            seq.metro_ids.astype(np.int64),
            seq.state_ids.astype(np.int64),
            seq.device_ids.astype(np.int64),
            label,
            user_id,
        )


def build_user_day_sequences(
    df: "pd.DataFrame",
    numeric_cols: List[str],
    max_seq_len: int,
) -> Dict[str, DailySequenceData]:
    """Convert a per-day dataframe into per-user day sequences (tail-truncated)."""
    sequences: Dict[str, DailySequenceData] = {}
    grouped = df.sort_values(["userId", "day"]).groupby("userId", sort=False)
    for user_id, g in tqdm(grouped, total=grouped.ngroups, desc="build_user_day_sequences"):
        g = g.tail(max_seq_len)
        sequences[user_id] = DailySequenceData(
            numeric=g[numeric_cols].to_numpy(np.float32),
            dow_ids=g["dow_id"].to_numpy(np.int64),
            metro_ids=g["metro_id"].to_numpy(np.int64),
            state_ids=g["state_id"].to_numpy(np.int64),
            device_ids=g["device_id"].to_numpy(np.int64),
        )
    return sequences


def collate_day_batch(batch):
    numeric_seq, dow_ids, metros, states, devices, labels, user_ids = zip(*batch)
    batch_size = len(batch)
    seq_lens = [len(x) for x in numeric_seq]
    max_len = max(seq_lens)
    num_feat_dim = numeric_seq[0].shape[1]

    num_tensor = torch.zeros(batch_size, max_len, num_feat_dim, dtype=torch.float32)
    dow_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    metro_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    state_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    device_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, (n, dow, m, s, d) in enumerate(zip(numeric_seq, dow_ids, metros, states, devices)):
        length = len(n)
        num_tensor[i, :length] = torch.from_numpy(n)
        dow_tensor[i, :length] = torch.from_numpy(dow)
        metro_tensor[i, :length] = torch.from_numpy(m)
        state_tensor[i, :length] = torch.from_numpy(s)
        device_tensor[i, :length] = torch.from_numpy(d)
        padding_mask[i, :length] = False

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return (
        num_tensor,
        dow_tensor,
        metro_tensor,
        state_tensor,
        device_tensor,
        padding_mask,
        labels_tensor,
        list(user_ids),
    )


class DailyChurnTransformer(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        num_dow: int,
        num_metro: int,
        num_state: int,
        num_device: int,
        config: TrainingConfig,
    ):
        super().__init__()
        self.config = config

        pooling = str(config.pooling).lower()
        if pooling not in {"mean", "attn"}:
            raise ValueError(f"Unknown pooling={config.pooling!r}; use 'mean' or 'attn'.")
        self.pooling = pooling

        self.dow_emb = nn.Embedding(num_dow + 1, 8, padding_idx=0)
        self.metro_emb = nn.Embedding(num_metro + 1, 16, padding_idx=0)
        self.state_emb = nn.Embedding(num_state + 1, 12, padding_idx=0)
        self.device_emb = nn.Embedding(num_device + 1, 6, padding_idx=0)

        cat_dim = 8 + 16 + 12 + 6
        self.input_proj = nn.Linear(num_numeric + cat_dim, config.d_model)
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
        self.attn_pool = nn.Linear(config.d_model, 1) if self.pooling == "attn" else None

        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self,
        numeric_feats: torch.Tensor,
        dow_ids: torch.Tensor,
        metro_ids: torch.Tensor,
        state_ids: torch.Tensor,
        device_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        cat_emb = torch.cat(
            [
                self.dow_emb(dow_ids),
                self.metro_emb(metro_ids),
                self.state_emb(state_ids),
                self.device_emb(device_ids),
            ],
            dim=-1,
        )
        x = torch.cat([numeric_feats, cat_emb], dim=-1)
        x = self.input_proj(x)
        x = self.input_dropout(x)
        x = self.pos_encoder(x)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        if self.pooling == "mean":
            mask = (~padding_mask).unsqueeze(-1)
            encoded = encoded * mask
            pooled = encoded.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            scores = self.attn_pool(encoded).squeeze(-1)
            scores = scores.masked_fill(padding_mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            weights = torch.nan_to_num(weights, nan=0.0).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1)

        logits = self.classifier(pooled).squeeze(-1)
        return logits


def make_day_dataloaders(
    train_sequences: Dict[str, DailySequenceData],
    val_sequences: Dict[str, DailySequenceData],
    labels: Dict[str, int],
    config: TrainingConfig,
):
    train_dataset = UserDaySequenceDataset(train_sequences, labels, list(train_sequences.keys()))
    val_dataset = UserDaySequenceDataset(val_sequences, labels, list(val_sequences.keys()))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_day_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_day_batch,
    )
    return train_loader, val_loader


def train_day_model(
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
) -> Tuple[List[float], List[float], nn.Module, int, Dict[str, torch.Tensor]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=lr * eta_min_factor
        )
        if use_cosine_decay
        else None
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")
    best_val = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = -1
    train_history: List[float] = []
    val_history: List[float] = []
    no_improve = 0

    for epoch in tqdm(range(epochs), desc="train_day_model"):
        model.train()
        running = 0.0
        for batch in train_loader:
            (
                num_feats,
                dow_ids,
                metro_ids,
                state_ids,
                device_ids,
                padding_mask,
                labels,
                _,
            ) = batch
            num_feats = num_feats.to(device)
            dow_ids = dow_ids.to(device)
            metro_ids = metro_ids.to(device)
            state_ids = state_ids.to(device)
            device_ids = device_ids.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(num_feats, dow_ids, metro_ids, state_ids, device_ids, padding_mask)
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
        with torch.no_grad():
            for batch in val_loader:
                (
                    num_feats,
                    dow_ids,
                    metro_ids,
                    state_ids,
                    device_ids,
                    padding_mask,
                    labels,
                    _,
                ) = batch
                num_feats = num_feats.to(device)
                dow_ids = dow_ids.to(device)
                metro_ids = metro_ids.to(device)
                state_ids = state_ids.to(device)
                device_ids = device_ids.to(device)
                padding_mask = padding_mask.to(device)
                labels = labels.to(device)

                logits = model(num_feats, dow_ids, metro_ids, state_ids, device_ids, padding_mask)
                if use_focal_loss:
                    probs = torch.sigmoid(logits)
                    pt = probs * labels + (1 - probs) * (1 - labels)
                    loss = bce(logits, labels) * torch.pow(1 - pt, focal_gamma)
                    loss = loss.mean()
                else:
                    loss = bce(logits, labels).mean()
                val_running += loss.item() * labels.size(0)
        val_loss = val_running / len(val_loader.dataset)
        val_history.append(val_loss)

        if val_loss < best_val - early_stop_min_delta:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
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


def predict_day_proba(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    probs: Dict[str, float] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict_day_proba"):
            (
                num_feats,
                dow_ids,
                metro_ids,
                state_ids,
                device_ids,
                padding_mask,
                _,
                user_ids,
            ) = batch
            num_feats = num_feats.to(device)
            dow_ids = dow_ids.to(device)
            metro_ids = metro_ids.to(device)
            state_ids = state_ids.to(device)
            device_ids = device_ids.to(device)
            padding_mask = padding_mask.to(device)

            logits = model(num_feats, dow_ids, metro_ids, state_ids, device_ids, padding_mask)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            for uid, prob in zip(user_ids, batch_probs):
                probs[uid] = float(prob)
    return probs


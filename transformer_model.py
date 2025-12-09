"""
Transformer model utilities for churn prediction on sequential user data.

ref: Attention is All You Need , Google
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

@dataclass
class TrainingConfig:
    d_model: int = 128  # Hidden size for Transformer projections/outputs
    nhead: int = 4  # Number of attention heads per encoder layer
    num_layers: int = 2  # Stacked Transformer encoder layer count
    dim_feedforward: int = 256  # Width of the feedforward block inside encoder layers
    dropout: float = 0.2  # Dropout rate applied to inputs and MLP
    max_seq_len: int = 400  # Maximum sequence length to retain per user
    batch_size: int = 32  # Mini-batch size for DataLoader
    lr: float = 1e-3  # Initial learning rate for AdamW
    weight_decay: float = 5e-4  # L2-style weight decay for regularization
    epochs: int = 8  # Number of training epochs
    num_workers: int = 0  # DataLoader worker processes for CPU-side batching
    use_cosine_decay: bool = False  # Whether to apply cosine annealing after warmup
    eta_min_factor: float = 0.1  # Multiplier for minimum LR in cosine schedule
    warmup_epochs: int = 0  # Linear warmup duration before decay starts
    use_focal_loss: bool = False  # Enable focal loss to focus on hard positives/negatives
    focal_gamma: float = 2.0  # Gamma parameter for focal loss curvature
    early_stop_patience: int = 0  # <=0 disables early stopping on validation loss
    early_stop_min_delta: float = 0.0  # Minimum improvement required to reset patience


@dataclass
class SequenceData:
    numeric: np.ndarray
    page_ids: np.ndarray
    metro_ids: np.ndarray
    state_ids: np.ndarray
    device_ids: np.ndarray


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
            seq.page_ids.astype(np.int64),
            seq.metro_ids.astype(np.int64),
            seq.state_ids.astype(np.int64),
            seq.device_ids.astype(np.int64),
            label,
            user_id,
        )


def build_user_sequences(
    df: "pd.DataFrame",
    numeric_cols: List[str],
    max_seq_len: int,
) -> Dict[str, SequenceData]:
    """
    Convert a per-event dataframe into per-user sequences truncated to the most
    recent `max_seq_len` steps.
    """
    sequences: Dict[str, SequenceData] = {}
    grouped = df.sort_values(["userId", "time"]).groupby("userId", sort=False)
    for user_id, g in grouped:
        g = g.tail(max_seq_len)
        sequences[user_id] = SequenceData(
            numeric=g[numeric_cols].to_numpy(np.float32),
            page_ids=g["page_id"].to_numpy(np.int64),
            metro_ids=g["metro_id"].to_numpy(np.int64),
            state_ids=g["state_id"].to_numpy(np.int64),
            device_ids=g["device_id"].to_numpy(np.int64),
        )
    return sequences


def collate_batch(batch):
    numeric_seq, pages, metros, states, devices, labels, user_ids = zip(*batch)
    batch_size = len(batch)
    seq_lens = [len(x) for x in numeric_seq]
    max_len = max(seq_lens)
    num_feat_dim = numeric_seq[0].shape[1]

    num_tensor = torch.zeros(batch_size, max_len, num_feat_dim, dtype=torch.float32)
    page_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    metro_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    state_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    device_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, (n, p, m, s, d) in enumerate(zip(numeric_seq, pages, metros, states, devices)):
        length = len(n)
        num_tensor[i, :length] = torch.from_numpy(n)
        page_tensor[i, :length] = torch.from_numpy(p)
        metro_tensor[i, :length] = torch.from_numpy(m)
        state_tensor[i, :length] = torch.from_numpy(s)
        device_tensor[i, :length] = torch.from_numpy(d)
        padding_mask[i, :length] = False  # False = keep, True = pad

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return (
        num_tensor,
        page_tensor,
        metro_tensor,
        state_tensor,
        device_tensor,
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


class ChurnTransformer(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        num_pages: int,
        num_metro: int,
        num_state: int,
        num_device: int,
        config: TrainingConfig,
    ):
        super().__init__()
        self.config = config
        self.page_emb = nn.Embedding(num_pages + 1, 32, padding_idx=0)
        self.metro_emb = nn.Embedding(num_metro + 1, 16, padding_idx=0)
        self.state_emb = nn.Embedding(num_state + 1, 12, padding_idx=0)
        self.device_emb = nn.Embedding(num_device + 1, 6, padding_idx=0)

        cat_dim = 32 + 16 + 12 + 6
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
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self,
        numeric_feats: torch.Tensor,
        page_ids: torch.Tensor,
        metro_ids: torch.Tensor,
        state_ids: torch.Tensor,
        device_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        cat_emb = torch.cat(
            [
                self.page_emb(page_ids),
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

        mask = (~padding_mask).unsqueeze(-1)
        encoded = encoded * mask
        pooled = encoded.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def make_dataloaders(
    train_sequences: Dict[str, SequenceData],
    val_sequences: Dict[str, SequenceData],
    labels: Dict[str, int],
    config: TrainingConfig,
):
    train_dataset = UserSequenceDataset(train_sequences, labels, list(train_sequences.keys()))
    val_dataset = UserSequenceDataset(val_sequences, labels, list(val_sequences.keys()))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader


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

    for epoch in tqdm(range(epochs)):
        model.train()
        running = 0.0
        for batch in train_loader:
            (
                num_feats,
                page_ids,
                metro_ids,
                state_ids,
                device_ids,
                padding_mask,
                labels,
                _,
            ) = batch
            num_feats = num_feats.to(device)
            page_ids = page_ids.to(device)
            metro_ids = metro_ids.to(device)
            state_ids = state_ids.to(device)
            device_ids = device_ids.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(num_feats, page_ids, metro_ids, state_ids, device_ids, padding_mask)
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
                    page_ids,
                    metro_ids,
                    state_ids,
                    device_ids,
                    padding_mask,
                    labels,
                    _,
                ) = batch
                num_feats = num_feats.to(device)
                page_ids = page_ids.to(device)
                metro_ids = metro_ids.to(device)
                state_ids = state_ids.to(device)
                device_ids = device_ids.to(device)
                padding_mask = padding_mask.to(device)
                labels = labels.to(device)

                logits = model(
                    num_feats, page_ids, metro_ids, state_ids, device_ids, padding_mask
                )
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


def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    probs: Dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            (
                num_feats,
                page_ids,
                metro_ids,
                state_ids,
                device_ids,
                padding_mask,
                _,
                user_ids,
            ) = batch
            num_feats = num_feats.to(device)
            page_ids = page_ids.to(device)
            metro_ids = metro_ids.to(device)
            state_ids = state_ids.to(device)
            device_ids = device_ids.to(device)
            padding_mask = padding_mask.to(device)
            logits = model(num_feats, page_ids, metro_ids, state_ids, device_ids, padding_mask)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            for uid, prob in zip(user_ids, batch_probs):
                probs[uid] = float(prob)
    return probs

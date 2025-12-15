"""Hybrid ResNet + Transformer model for churn prediction on user-day sequences.

Intended usage:
- Input: numeric daily features shaped [B, T, F]
- Local extractor: 1D ResNet blocks over time axis (Conv1d)
- Global model: Transformer encoder
- Head: masked mean pooling + MLP classifier

This is designed to plug into the existing training utilities in
`churn_pipeline.transformer_user_day` (same forward signature).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from churn_pipeline.transformer_user_day import PositionalEncoding, TrainingConfig


class ResNet1DBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 5,
        dropout: float = 0.1,
        use_gelu: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd for same-length padding")
        padding = kernel_size // 2

        act: nn.Module = nn.GELU() if use_gelu else nn.ReLU()

        # GroupNorm(1, C) behaves similarly to LayerNorm over channels and is
        # typically more stable than BatchNorm for small batch sizes.
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(1, channels),
            act,
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(1, channels),
        )
        self.out_act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return self.out_act(x + self.block(x))


class ResNetStem(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        depth: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResNet1DBlock(channels, kernel_size=kernel_size, dropout=dropout)
                for _ in range(int(depth))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        for blk in self.blocks:
            x = blk(x)
        return x


class ChurnResNetTransformerUserDay(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        config: TrainingConfig,
        *,
        resnet_depth: int = 3,
        resnet_kernel_size: int = 5,
        resnet_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(num_numeric, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)

        stem_dropout = float(config.dropout) if resnet_dropout is None else float(resnet_dropout)
        self.resnet = ResNetStem(
            channels=config.d_model,
            depth=int(resnet_depth),
            kernel_size=int(resnet_kernel_size),
            dropout=stem_dropout,
        )

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
        # numeric_feats: [B, T, F]
        # padding_mask: [B, T]  True=pad, False=keep
        x = self.input_proj(numeric_feats)
        x = self.input_dropout(x)

        # Local extractor over time axis with Conv1d (channels-first)
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.resnet(x)
        x = x.transpose(1, 2)  # [B, T, C]

        # Optional: keep padded tokens as zeros before transformer
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = self.pos_encoder(x)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        mask = (~padding_mask).unsqueeze(-1)
        encoded = encoded * mask
        pooled = encoded.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

from __future__ import annotations

import torch
from torch import nn


class GlobalBiGRUScorer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        token_emb_dim: int = 256,
        row_emb_dim: int = 16,
        col_emb_dim: int = 16,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_rows: int = 4,
        num_cols: int = 7,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size
        self.token_emb = nn.Embedding(vocab_size + 1, token_emb_dim)
        self.row_emb = nn.Embedding(num_rows, row_emb_dim)
        self.col_emb = nn.Embedding(num_cols, col_emb_dim)
        input_dim = token_emb_dim + row_emb_dim + col_emb_dim
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                self.token_emb(input_ids),
                self.row_emb(row_ids),
                self.col_emb(col_ids),
            ],
            dim=-1,
        )
        hidden, _ = self.encoder(x)
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        target_hidden = hidden[batch_indices, target_pos]
        target_hidden = self.dropout(self.norm(target_hidden))
        return self.head(target_hidden)

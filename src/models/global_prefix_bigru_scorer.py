from __future__ import annotations

import torch
from torch import nn


class GlobalPrefixBiGRUScorer(nn.Module):
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
        same_row_weight: float = 1.0,
        adjacent_row_weight: float = 0.45,
        far_row_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size
        self.prefix_pad_id = vocab_size + 1
        self.col_pad_id = vocab_size + 2
        self.same_row_weight = same_row_weight
        self.adjacent_row_weight = adjacent_row_weight
        self.far_row_weight = far_row_weight

        self.token_emb = nn.Embedding(vocab_size + 3, token_emb_dim)
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
        self.norm = nn.LayerNorm(hidden_size * 4)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 4, vocab_size)

    def context_weights(
        self,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = row_ids.shape
        batch_indices = torch.arange(batch_size, device=row_ids.device)
        target_rows = row_ids[batch_indices, target_pos].unsqueeze(1)
        target_cols = col_ids[batch_indices, target_pos].unsqueeze(1)
        row_distance = (row_ids - target_rows).abs()
        col_distance = (col_ids - target_cols).abs().float()

        weights = torch.full((batch_size, seq_len), self.far_row_weight, device=row_ids.device)
        weights = torch.where(row_distance == 1, torch.full_like(weights, self.adjacent_row_weight), weights)
        same_line_decay = 1.0 / (1.0 + col_distance)
        weights = torch.where(row_distance == 0, self.same_row_weight * same_line_decay, weights)
        weights[batch_indices, target_pos] = 0.0
        return weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

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
        weights = self.context_weights(row_ids, col_ids, target_pos)
        context_hidden = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        combined = torch.cat([target_hidden, context_hidden], dim=-1)
        return self.head(self.dropout(self.norm(combined)))

from __future__ import annotations

import torch
from torch import nn


class CharPoemLM(nn.Module):
    """A compact character LM with explicit positional information."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 256,
        pos_dim: int = 32,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_len: int = 64,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()
        if rnn_type.lower() != "gru":
            raise ValueError("This final project version keeps only GRU checkpoints.")
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_len, pos_dim)
        self.rnn = nn.GRU(
            input_size=emb_dim + pos_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len={self.max_len}")
        pos = torch.arange(seq_len, device=input_ids.device).expand(batch_size, seq_len)
        x = torch.cat([self.token_emb(input_ids), self.pos_emb(pos)], dim=-1)
        hidden, _ = self.rnn(x)
        return self.head(self.norm(hidden))

    def step(self, token_ids: torch.Tensor, position: int, state=None):
        """Run one autoregressive step while reusing the GRU hidden state."""
        if position >= self.max_len:
            raise ValueError(f"position {position} exceeds max_len={self.max_len}")
        pos = torch.full_like(token_ids, position)
        x = torch.cat([self.token_emb(token_ids), self.pos_emb(pos)], dim=-1)
        hidden, state = self.rnn(x, state)
        logits = self.head(self.norm(hidden[:, -1]))
        return logits, state

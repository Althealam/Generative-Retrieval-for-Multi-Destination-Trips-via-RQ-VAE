import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TrueRQVAETransformer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 128,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        max_len: int = 256,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.pad_token = codebook_size
        self.num_codes = codebook_size + 1
        self.d_model = d_model

        self.embedding = nn.Embedding(self.num_codes, d_model, padding_idx=self.pad_token)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_block = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc_code1 = nn.Linear(d_model, codebook_size)
        self.fc_code2 = nn.Linear(d_model, codebook_size)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, x: torch.Tensor):
        padding_mask = x.eq(self.pad_token)
        causal_mask = self._generate_causal_mask(x.size(1), x.device)
        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)
        h = self.transformer_block(h, mask=causal_mask, src_key_padding_mask=padding_mask)
        last_hidden = h[:, -1, :]
        return self.fc_code1(last_hidden), self.fc_code2(last_hidden)

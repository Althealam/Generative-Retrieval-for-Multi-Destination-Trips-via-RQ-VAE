import math

import torch
import torch.nn as nn

from src.models.embedding.positional import PositionalEncoding


class CityTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 256,
        n_booker_countries: int = 0,
        n_device_classes: int = 0,
        pooling: str = "last",
    ):
        super().__init__()
        if pooling not in {"last", "mean", "cls"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.pooling = pooling
        self.vocab_size = vocab_size
        self.cls_token_id = vocab_size

        # Reserve one extra input token for optional CLS pooling.
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.emb_booker = nn.Embedding(n_booker_countries + 1, 64, padding_idx=0)
        self.emb_device = nn.Embedding(n_device_classes + 1, 48, padding_idx=0)
        self.emb_month = nn.Embedding(13, 32)
        self.emb_stay = nn.Embedding(31, 48)
        self.emb_trip_len = nn.Embedding(31, 32)
        self.emb_num_unique = nn.Embedding(31, 32)
        self.emb_repeat_ratio = nn.Embedding(11, 24)
        self.emb_last_stay = nn.Embedding(31, 32)
        self.emb_same_country_streak = nn.Embedding(31, 32)
        ctx_dim = 64 + 48 + 32 + 48 + 32 + 32 + 24 + 32 + 32
        self.ctx_proj = nn.Linear(ctx_dim, d_model)

        self.classifier = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        booker_idx: torch.Tensor,
        device_idx: torch.Tensor,
        month_idx: torch.Tensor,
        stay_idx: torch.Tensor,
        trip_len_idx: torch.Tensor,
        num_unique_idx: torch.Tensor,
        repeat_ratio_idx: torch.Tensor,
        last_stay_idx: torch.Tensor,
        same_country_streak_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling == "cls":
            cls_col = torch.full(
                (x.size(0), 1),
                fill_value=self.cls_token_id,
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([cls_col, x], dim=1)

        padding_mask = x.eq(self.pad_token_id)
        causal_mask = self._generate_causal_mask(x.size(1), x.device)

        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)
        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=padding_mask)
        if self.pooling == "last":
            valid_lengths = (~padding_mask).sum(dim=1).clamp(min=1)
            last_indices = (valid_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, h.size(2))
            seq_hidden = h.gather(dim=1, index=last_indices).squeeze(1)
        elif self.pooling == "mean":
            valid_mask = (~padding_mask).unsqueeze(-1).to(h.dtype)
            masked_h = h * valid_mask
            valid_counts = valid_mask.sum(dim=1).clamp(min=1.0)
            seq_hidden = masked_h.sum(dim=1) / valid_counts
        else:  # cls
            seq_hidden = h[:, 0, :]

        ctx = torch.cat(
            [
                self.emb_booker(booker_idx),
                self.emb_device(device_idx),
                self.emb_month(month_idx),
                self.emb_stay(stay_idx),
                self.emb_trip_len(trip_len_idx),
                self.emb_num_unique(num_unique_idx),
                self.emb_repeat_ratio(repeat_ratio_idx),
                self.emb_last_stay(last_stay_idx),
                self.emb_same_country_streak(same_country_streak_idx),
            ],
            dim=-1,
        )
        seq_hidden = seq_hidden + self.ctx_proj(ctx)

        return self.classifier(seq_hidden)

import math

import torch
import torch.nn as nn

from src.models.embedding.positional import PositionalEncoding


class RQVAETransformer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 128,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        max_len: int = 256,
        n_booker_countries: int = 0,
        n_device_classes: int = 0,
        n_hotel_countries: int = 0,
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
        self.emb_booker = nn.Embedding(n_booker_countries + 1, 64, padding_idx=0)
        self.emb_device = nn.Embedding(n_device_classes + 1, 48, padding_idx=0)
        self.emb_month = nn.Embedding(13, 32)
        self.emb_stay = nn.Embedding(31, 48)
        self.emb_trip_len = nn.Embedding(31, 32)
        self.emb_num_unique = nn.Embedding(31, 32)
        self.emb_repeat_ratio = nn.Embedding(11, 24)
        self.emb_last_stay = nn.Embedding(31, 32)
        self.emb_same_country_streak = nn.Embedding(31, 32)
        self.emb_last_hotel_country = nn.Embedding(n_hotel_countries + 1, 64, padding_idx=0)
        self.emb_unique_hotel_countries = nn.Embedding(31, 32)
        self.emb_cross_border_count = nn.Embedding(31, 32)
        self.emb_cross_border_ratio = nn.Embedding(11, 24)
        self.ctx_proj = nn.Linear(64 + 48 + 32 + 48 + 32 + 32 + 24 + 32 + 32 + 64 + 32 + 32 + 24, d_model)
        self.fc_code1 = nn.Linear(d_model, codebook_size)
        self.fc_code2 = nn.Linear(d_model, codebook_size)

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
        last_hotel_country_idx: torch.Tensor,
        unique_hotel_countries_idx: torch.Tensor,
        cross_border_count_idx: torch.Tensor,
        cross_border_ratio_idx: torch.Tensor,
    ):
        padding_mask = x.eq(self.pad_token)
        causal_mask = self._generate_causal_mask(x.size(1), x.device)
        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)
        h = self.transformer_block(h, mask=causal_mask, src_key_padding_mask=padding_mask)
        valid_lengths = (~padding_mask).sum(dim=1).clamp(min=1)
        last_indices = (valid_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, h.size(2))
        last_hidden = h.gather(dim=1, index=last_indices).squeeze(1)
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
                self.emb_last_hotel_country(last_hotel_country_idx),
                self.emb_unique_hotel_countries(unique_hotel_countries_idx),
                self.emb_cross_border_count(cross_border_count_idx),
                self.emb_cross_border_ratio(cross_border_ratio_idx),
            ],
            dim=-1,
        )
        last_hidden = last_hidden + self.ctx_proj(ctx)
        return self.fc_code1(last_hidden), self.fc_code2(last_hidden)

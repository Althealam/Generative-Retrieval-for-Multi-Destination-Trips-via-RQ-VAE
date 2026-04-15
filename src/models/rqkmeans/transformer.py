import math

import torch
import torch.nn as nn

from src.models.embedding.positional import PositionalEncoding


class RQKMeansTransformer(nn.Module):
    def __init__(
        self,
        num_codes: int = 33,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_len: int = 100,
        codebook_size: int = 32,
        pad_code: int = 32,
        n_booker_countries: int = 0,
        n_device_classes: int = 0,
        n_hotel_countries: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_code = pad_code
        self.codebook_size = codebook_size

        self.embedding = nn.Embedding(num_codes, d_model, padding_idx=pad_code)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_block = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

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

    def _generate_causal_mask(self, sz: int, device):
        return torch.triu(
            torch.ones(sz, sz, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        x,
        booker_idx,
        device_idx,
        month_idx,
        stay_idx,
        trip_len_idx,
        num_unique_idx,
        repeat_ratio_idx,
        last_stay_idx,
        same_country_streak_idx,
        last_hotel_country_idx,
        unique_hotel_countries_idx,
        cross_border_count_idx,
        cross_border_ratio_idx,
    ):
        _, seq_len = x.size()
        device = x.device

        padding_mask = x == self.pad_code
        causal_mask = self._generate_causal_mask(seq_len, device)

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.transformer_block(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        valid_lengths = (~padding_mask).sum(dim=1).clamp(min=1)
        last_indices = (valid_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, output.size(2))
        last_hidden = output.gather(dim=1, index=last_indices).squeeze(1)
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

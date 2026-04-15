import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CityGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        n_booker_countries: int = 0,
        n_device_classes: int = 0,
        n_hotel_countries: int = 0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

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
        self.ctx_proj = nn.Linear(ctx_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, vocab_size)

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
    ) -> torch.Tensor:
        lengths = x.ne(self.pad_token_id).sum(dim=1).clamp(min=1).cpu()
        embeds = self.embedding(x)
        packed = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        _, hn = self.gru(packed)
        last_hidden = hn[-1]

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
        return self.classifier(last_hidden)

import torch
import torch.nn as nn


class RQKmeansGRU(nn.Module):
    """GRU baseline over code tokens + same trip context fusion as the RQ-KMeans Transformer."""

    def __init__(
        self,
        num_codes: int = 33,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        codebook_size: int = 32,
        n_booker_countries: int = 0,
        n_device_classes: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, embedding_dim, padding_idx=32)
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

        self.fc_code1 = nn.Linear(hidden_dim, codebook_size)
        self.fc_code2 = nn.Linear(hidden_dim, codebook_size)

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
    ):
        embeds = self.embedding(x)
        _, hn = self.gru(embeds)
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
            ],
            dim=-1,
        )
        last_hidden = last_hidden + self.ctx_proj(ctx)
        return self.fc_code1(last_hidden), self.fc_code2(last_hidden)

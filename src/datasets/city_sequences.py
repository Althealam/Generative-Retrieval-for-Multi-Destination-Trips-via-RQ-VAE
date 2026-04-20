from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.datasets.tokens import PAD_TOKEN_ID


class CitySequenceDataset(Dataset):
    def __init__(
        self,
        x_values: list[list[int]],
        y_values: list[int] | None = None,
        *,
        ctx_booker: list[int],
        ctx_device: list[int],
        ctx_affiliate: list[int],
        ctx_month: list[int],
        ctx_stay: list[int],
        ctx_trip_len: list[int],
        ctx_num_unique_cities: list[int],
        ctx_repeat_city_ratio: list[int],
        ctx_last_stay_days: list[int],
        ctx_same_country_streak: list[int],
        ctx_last_hotel_country: list[int],
        ctx_unique_hotel_countries: list[int],
        ctx_cross_border_count: list[int],
        ctx_cross_border_ratio: list[int],
        ctx_sem_code1: list[int],
        ctx_sem_code2: list[int],
    ):
        self.x_values = [torch.tensor(x, dtype=torch.long) for x in x_values]
        self.y_values = torch.tensor(y_values, dtype=torch.long) if y_values is not None else None
        self.ctx_booker = torch.tensor(ctx_booker, dtype=torch.long)
        self.ctx_device = torch.tensor(ctx_device, dtype=torch.long)
        self.ctx_affiliate = torch.tensor(ctx_affiliate, dtype=torch.long)
        self.ctx_month = torch.tensor(ctx_month, dtype=torch.long)
        self.ctx_stay = torch.tensor(ctx_stay, dtype=torch.long)
        self.ctx_trip_len = torch.tensor(ctx_trip_len, dtype=torch.long)
        self.ctx_num_unique_cities = torch.tensor(ctx_num_unique_cities, dtype=torch.long)
        self.ctx_repeat_city_ratio = torch.tensor(ctx_repeat_city_ratio, dtype=torch.long)
        self.ctx_last_stay_days = torch.tensor(ctx_last_stay_days, dtype=torch.long)
        self.ctx_same_country_streak = torch.tensor(ctx_same_country_streak, dtype=torch.long)
        self.ctx_last_hotel_country = torch.tensor(ctx_last_hotel_country, dtype=torch.long)
        self.ctx_unique_hotel_countries = torch.tensor(ctx_unique_hotel_countries, dtype=torch.long)
        self.ctx_cross_border_count = torch.tensor(ctx_cross_border_count, dtype=torch.long)
        self.ctx_cross_border_ratio = torch.tensor(ctx_cross_border_ratio, dtype=torch.long)
        self.ctx_sem_code1 = torch.tensor(ctx_sem_code1, dtype=torch.long)
        self.ctx_sem_code2 = torch.tensor(ctx_sem_code2, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x_values)

    def __getitem__(self, idx: int):
        if self.y_values is None:
            return (
                self.x_values[idx],
                self.ctx_booker[idx],
                self.ctx_device[idx],
                self.ctx_affiliate[idx],
                self.ctx_month[idx],
                self.ctx_stay[idx],
                self.ctx_trip_len[idx],
                self.ctx_num_unique_cities[idx],
                self.ctx_repeat_city_ratio[idx],
                self.ctx_last_stay_days[idx],
                self.ctx_same_country_streak[idx],
                self.ctx_last_hotel_country[idx],
                self.ctx_unique_hotel_countries[idx],
                self.ctx_cross_border_count[idx],
                self.ctx_cross_border_ratio[idx],
                self.ctx_sem_code1[idx],
                self.ctx_sem_code2[idx],
            )
        return (
            self.x_values[idx],
            self.y_values[idx],
            self.ctx_booker[idx],
            self.ctx_device[idx],
            self.ctx_affiliate[idx],
            self.ctx_month[idx],
            self.ctx_stay[idx],
            self.ctx_trip_len[idx],
            self.ctx_num_unique_cities[idx],
            self.ctx_repeat_city_ratio[idx],
            self.ctx_last_stay_days[idx],
            self.ctx_same_country_streak[idx],
            self.ctx_last_hotel_country[idx],
            self.ctx_unique_hotel_countries[idx],
            self.ctx_cross_border_count[idx],
            self.ctx_cross_border_ratio[idx],
            self.ctx_sem_code1[idx],
            self.ctx_sem_code2[idx],
        )


def collate_city_batch(batch):
    n_fields = len(batch[0])
    if n_fields == 17:
        xs, bs, ds, afs, ms, ss, tls, nus, rrs, lss, scs, lcs, ucs, bcs, brs, sem1s, sem2s = zip(*batch)
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=PAD_TOKEN_ID)
        return (
            xs_padded,
            torch.stack(bs),
            torch.stack(ds),
            torch.stack(afs),
            torch.stack(ms),
            torch.stack(ss),
            torch.stack(tls),
            torch.stack(nus),
            torch.stack(rrs),
            torch.stack(lss),
            torch.stack(scs),
            torch.stack(lcs),
            torch.stack(ucs),
            torch.stack(bcs),
            torch.stack(brs),
            torch.stack(sem1s),
            torch.stack(sem2s),
        )
    if n_fields == 18:
        xs, ys, bs, ds, afs, ms, ss, tls, nus, rrs, lss, scs, lcs, ucs, bcs, brs, sem1s, sem2s = zip(*batch)
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=PAD_TOKEN_ID)
        return (
            xs_padded,
            torch.stack(ys),
            torch.stack(bs),
            torch.stack(ds),
            torch.stack(afs),
            torch.stack(ms),
            torch.stack(ss),
            torch.stack(tls),
            torch.stack(nus),
            torch.stack(rrs),
            torch.stack(lss),
            torch.stack(scs),
            torch.stack(lcs),
            torch.stack(ucs),
            torch.stack(bcs),
            torch.stack(brs),
            torch.stack(sem1s),
            torch.stack(sem2s),
        )
    raise ValueError(f"Unexpected batch tuple length {n_fields}")


def build_city_dataloaders(
    train_x: list[list[int]],
    train_y: list[int],
    test_x: list[list[int]],
    batch_size: int = 256,
    *,
    train_ctx: tuple[
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
    ],
    test_ctx: tuple[
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
    ],
) -> tuple[DataLoader, DataLoader]:
    tb, td, ta, tm, ts, ttl, tnu, trr, tls, tsc, tlc, tuc, tbc, tbr, tsem1, tsem2 = train_ctx
    eb, ed, ea, em, es, etl, enu, err, els, esc, elc, euc, ebc, ebr, esem1, esem2 = test_ctx
    train_ds = CitySequenceDataset(
        train_x,
        train_y,
        ctx_booker=tb,
        ctx_device=td,
        ctx_affiliate=ta,
        ctx_month=tm,
        ctx_stay=ts,
        ctx_trip_len=ttl,
        ctx_num_unique_cities=tnu,
        ctx_repeat_city_ratio=trr,
        ctx_last_stay_days=tls,
        ctx_same_country_streak=tsc,
        ctx_last_hotel_country=tlc,
        ctx_unique_hotel_countries=tuc,
        ctx_cross_border_count=tbc,
        ctx_cross_border_ratio=tbr,
        ctx_sem_code1=tsem1,
        ctx_sem_code2=tsem2,
    )
    test_ds = CitySequenceDataset(
        test_x,
        ctx_booker=eb,
        ctx_device=ed,
        ctx_affiliate=ea,
        ctx_month=em,
        ctx_stay=es,
        ctx_trip_len=etl,
        ctx_num_unique_cities=enu,
        ctx_repeat_city_ratio=err,
        ctx_last_stay_days=els,
        ctx_same_country_streak=esc,
        ctx_last_hotel_country=elc,
        ctx_unique_hotel_countries=euc,
        ctx_cross_border_count=ebc,
        ctx_cross_border_ratio=ebr,
        ctx_sem_code1=esem1,
        ctx_sem_code2=esem2,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_city_batch
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_city_batch
    )
    return train_loader, test_loader

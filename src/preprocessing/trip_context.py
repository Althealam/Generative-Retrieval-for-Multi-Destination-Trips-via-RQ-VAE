"""Trip-level side features aligned with create_mutliple_sequences aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_booker_device_vocabs(train_trips: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], int, int]:
    countries = sorted(train_trips["booker_country"].dropna().astype(str).unique().tolist())
    devices = sorted(train_trips["device_class"].dropna().astype(str).unique().tolist())
    booker_to_idx = {c: i + 1 for i, c in enumerate(countries)}
    device_to_idx = {d: i + 1 for i, d in enumerate(devices)}
    return booker_to_idx, device_to_idx, len(countries), len(devices)


def _bucket_trip_len(n: int) -> int:
    return max(1, min(30, int(n)))


def _bucket_repeat_ratio(r: float) -> int:
    return max(1, min(10, int(r * 10) + 1))


def _compute_same_country_streak(countries: list[str]) -> int:
    if not countries:
        return 1
    last = countries[-1]
    streak = 1
    for i in range(len(countries) - 2, -1, -1):
        if countries[i] == last:
            streak += 1
        else:
            break
    return max(1, min(30, streak))


def row_to_context_indices(
    row: pd.Series, booker_to_idx: dict[str, int], device_to_idx: dict[str, int]
) -> tuple[int, int, int, int, int, int, int, int, int]:
    booker = booker_to_idx.get(str(row["booker_country"]), 0) if pd.notna(row["booker_country"]) else 0
    device = device_to_idx.get(str(row["device_class"]), 0) if pd.notna(row["device_class"]) else 0

    m = row["checkin_month"]
    if pd.isna(m):
        month_idx = 0
    else:
        mi = int(m)
        month_idx = mi if 1 <= mi <= 12 else 0

    durs = row["stay_duration"]
    if isinstance(durs, (list, np.ndarray)) and len(durs) > 0:
        mean_stay = int(round(float(np.mean(durs))))
    else:
        mean_stay = 1
    mean_stay = max(1, min(30, mean_stay))
    stay_idx = mean_stay

    cities = row["city_id"] if isinstance(row["city_id"], (list, np.ndarray)) else []
    trip_len = len(cities)
    num_unique = len(set(cities)) if trip_len > 0 else 1
    repeat_ratio = 1.0 - (num_unique / trip_len) if trip_len > 0 else 0.0

    trip_len_idx = _bucket_trip_len(trip_len if trip_len > 0 else 1)
    num_unique_idx = _bucket_trip_len(num_unique)
    repeat_ratio_idx = _bucket_repeat_ratio(repeat_ratio)

    if isinstance(durs, (list, np.ndarray)) and len(durs) > 0:
        last_stay = int(durs[-1])
    else:
        last_stay = 1
    last_stay_idx = max(1, min(30, last_stay))

    countries_raw = row["hotel_country"] if isinstance(row.get("hotel_country"), list) else []
    countries = [str(c) for c in countries_raw if pd.notna(c)]
    same_country_streak_idx = _compute_same_country_streak(countries)

    return (
        booker,
        device,
        month_idx,
        stay_idx,
        trip_len_idx,
        num_unique_idx,
        repeat_ratio_idx,
        last_stay_idx,
        same_country_streak_idx,
    )

"""Trip-level side features aligned with trip aggregates (booker, device, stay, etc.)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_booker_device_affiliate_vocabs(
    train_trips: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], int, int, int]:
    countries: set[str] = set()
    for values in train_trips["booker_country"].tolist():
        if isinstance(values, list):
            for country in values:
                if pd.notna(country):
                    countries.add(str(country))
        elif pd.notna(values):
            countries.add(str(values))
    devices: set[str] = set()
    for values in train_trips["device_class"].tolist():
        if isinstance(values, list):
            for device in values:
                if pd.notna(device):
                    devices.add(str(device))
        elif pd.notna(values):
            devices.add(str(values))
    affiliates: set[str] = set()
    for values in train_trips["affiliate_id"].tolist():
        if isinstance(values, list):
            for affiliate in values:
                if pd.notna(affiliate):
                    affiliates.add(str(affiliate))
        elif pd.notna(values):
            affiliates.add(str(values))
    booker_to_idx = {c: i + 1 for i, c in enumerate(sorted(countries))}
    device_to_idx = {d: i + 1 for i, d in enumerate(sorted(devices))}
    affiliate_to_idx = {a: i + 1 for i, a in enumerate(sorted(affiliates))}
    return booker_to_idx, device_to_idx, affiliate_to_idx, len(countries), len(devices), len(affiliates)


def build_booker_device_vocabs(train_trips: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], int, int]:
    booker_to_idx, device_to_idx, _affiliate_to_idx, n_booker, n_device, _n_affiliate = (
        build_booker_device_affiliate_vocabs(train_trips)
    )
    return booker_to_idx, device_to_idx, n_booker, n_device


def build_hotel_country_vocab(train_trips: pd.DataFrame) -> tuple[dict[str, int], int]:
    countries: set[str] = set()
    for values in train_trips["hotel_country"].tolist():
        if isinstance(values, list):
            for country in values:
                if pd.notna(country):
                    countries.add(str(country))
    sorted_countries = sorted(countries)
    country_to_idx = {country: i + 1 for i, country in enumerate(sorted_countries)}
    return country_to_idx, len(sorted_countries)


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
    row: pd.Series,
    booker_to_idx: dict[str, int],
    device_to_idx: dict[str, int],
    affiliate_to_idx: dict[str, int],
    *,
    prefix_len: int | None = None,
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    durs_raw = row["stay_duration"]
    if isinstance(durs_raw, (list, np.ndarray)):
        durs = list(durs_raw)
    else:
        durs = []

    cities_raw = row["city_id"] if isinstance(row["city_id"], (list, np.ndarray)) else []
    cities = list(cities_raw)

    countries_raw = row["hotel_country"] if isinstance(row.get("hotel_country"), list) else []
    countries = [str(c) for c in countries_raw if pd.notna(c)]
    bookers_raw = row["booker_country"] if isinstance(row.get("booker_country"), list) else []
    bookers = [str(b) for b in bookers_raw if pd.notna(b)]
    devices_raw = row["device_class"] if isinstance(row.get("device_class"), list) else []
    devices = [str(d) for d in devices_raw if pd.notna(d)]
    affiliates_raw = row["affiliate_id"] if isinstance(row.get("affiliate_id"), list) else []
    affiliates = [str(a) for a in affiliates_raw if pd.notna(a)]
    months_raw = row["checkin_month"] if isinstance(row.get("checkin_month"), list) else []
    months = [int(m) for m in months_raw if pd.notna(m)]

    # Prevent future leakage: use only prefix information for this sample.
    if prefix_len is None:
        prefix_len = len(cities)
    prefix_len = max(1, min(int(prefix_len), len(cities) if len(cities) > 0 else 1))

    cities_prefix = cities[:prefix_len]
    durs_prefix = durs[:prefix_len]
    countries_prefix = countries[:prefix_len]
    bookers_prefix = bookers[:prefix_len]
    devices_prefix = devices[:prefix_len]
    affiliates_prefix = affiliates[:prefix_len]
    months_prefix = months[:prefix_len]
    if bookers_prefix:
        booker = booker_to_idx.get(bookers_prefix[-1], 0)
    else:
        booker_raw = row.get("booker_country")
        booker = booker_to_idx.get(str(booker_raw), 0) if pd.notna(booker_raw) else 0
    if devices_prefix:
        device = device_to_idx.get(devices_prefix[-1], 0)
    else:
        device_raw = row.get("device_class")
        device = device_to_idx.get(str(device_raw), 0) if pd.notna(device_raw) else 0
    affiliate_idx = affiliate_to_idx.get(affiliates_prefix[-1], 0) if affiliates_prefix else 0
    if months_prefix:
        last_month = int(months_prefix[-1])
        month_idx = last_month if 1 <= last_month <= 12 else 0
    else:
        month_raw = row.get("checkin_month")
        if pd.isna(month_raw):
            month_idx = 0
        else:
            mi = int(month_raw)
            month_idx = mi if 1 <= mi <= 12 else 0

    if len(durs_prefix) > 0:
        mean_stay = int(round(float(np.mean(durs_prefix))))
    else:
        mean_stay = 1
    mean_stay = max(1, min(30, mean_stay))
    stay_idx = mean_stay

    trip_len = len(cities_prefix)
    num_unique = len(set(cities_prefix)) if trip_len > 0 else 1
    repeat_ratio = 1.0 - (num_unique / trip_len) if trip_len > 0 else 0.0

    trip_len_idx = _bucket_trip_len(trip_len if trip_len > 0 else 1)
    num_unique_idx = _bucket_trip_len(num_unique)
    repeat_ratio_idx = _bucket_repeat_ratio(repeat_ratio)

    if len(durs_prefix) > 0:
        last_stay = int(durs_prefix[-1])
    else:
        last_stay = 1
    last_stay_idx = max(1, min(30, last_stay))

    same_country_streak_idx = _compute_same_country_streak(countries_prefix)

    return (
        booker,
        device,
        affiliate_idx,
        month_idx,
        stay_idx,
        trip_len_idx,
        num_unique_idx,
        repeat_ratio_idx,
        last_stay_idx,
        same_country_streak_idx,
    )


def row_to_spatial_indices(
    row: pd.Series,
    hotel_country_to_idx: dict[str, int],
    *,
    prefix_len: int | None = None,
) -> tuple[int, int, int, int]:
    countries_raw = row["hotel_country"] if isinstance(row.get("hotel_country"), list) else []
    countries = [str(c) for c in countries_raw if pd.notna(c)]
    if prefix_len is None:
        prefix_len = len(countries)
    prefix_len = max(1, min(int(prefix_len), len(countries) if len(countries) > 0 else 1))
    countries_prefix = countries[:prefix_len]

    if countries_prefix:
        last_country_idx = hotel_country_to_idx.get(countries_prefix[-1], 0)
    else:
        last_country_idx = 0

    unique_country_count = len(set(countries_prefix)) if countries_prefix else 1
    unique_country_idx = _bucket_trip_len(unique_country_count)

    if len(countries_prefix) <= 1:
        cross_border_count = 0
    else:
        cross_border_count = sum(
            1 for i in range(1, len(countries_prefix)) if countries_prefix[i] != countries_prefix[i - 1]
        )
    cross_border_count_idx = max(0, min(30, int(cross_border_count)))

    denom = max(1, len(countries_prefix) - 1)
    cross_border_ratio = cross_border_count / denom
    cross_border_ratio_idx = max(0, min(10, int(round(cross_border_ratio * 10))))

    return (
        last_country_idx,
        unique_country_idx,
        cross_border_count_idx,
        cross_border_ratio_idx,
    )

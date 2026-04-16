"""Aggregate row-level bookings into one row per trip."""

from __future__ import annotations

import pandas as pd


def create_trip_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each `utrip_id` into a chronological city sequence."""
    data = df.copy()
    data["checkin"] = pd.to_datetime(data["checkin"])
    data = data.sort_values(["utrip_id", "checkin"])

    sequences = (
        data.groupby("utrip_id")
        .agg({"city_id": list, "hotel_country": "last", "booker_country": "first"})
        .reset_index()
    )
    return sequences


def create_multiple_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    聚合后每行主要有：city_id(list), stay_duration(list), checkin_month(list), booker_country(list), device_class(list)
    将原始数据变成trip级的数据
    """
    data = df.copy()
    data["checkin"] = pd.to_datetime(data["checkin"])
    data["checkout"] = pd.to_datetime(data["checkout"])
    data = data.sort_values(["utrip_id", "checkin"])

    data["stay_duration"] = (data["checkout"] - data["checkin"]).dt.days
    data["stay_duration"] = data["stay_duration"].clip(1, 30)
    data["checkin_month"] = data["checkin"].dt.month

    sequences = data.groupby("utrip_id").agg(
        {
            "city_id": list,
            "hotel_country": list,
            "affiliate_id": list,
            "device_class": list,
            "stay_duration": list,
            "checkin_month": list,
            "booker_country": list,
        }
    ).reset_index()

    return sequences

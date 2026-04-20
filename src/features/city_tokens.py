"""City-id token sequences and vocab for the embedding (CityTransformer) path."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.datasets.tokens import UNK_TOKEN_ID
from src.features.context import row_to_context_indices, row_to_spatial_indices


def _semantic_code_from_city(
    city_id: int | None,
    city_to_semantic_codes: dict[int, tuple[int, int]] | None,
) -> tuple[int, int]:
    if city_id is None or city_to_semantic_codes is None or city_id not in city_to_semantic_codes:
        return 0, 0
    sem1, sem2 = city_to_semantic_codes[city_id]
    # Reserve 0 for missing/padding semantic side information.
    return int(sem1) + 1, int(sem2) + 1


@dataclass
class CitySequencePack:
    x: list[list[int]]
    y: list[int] | None
    ctx_booker: list[int]
    ctx_device: list[int]
    ctx_affiliate: list[int]
    ctx_month: list[int]
    ctx_stay: list[int]
    ctx_trip_len: list[int]
    ctx_num_unique_cities: list[int]
    ctx_repeat_city_ratio: list[int]
    ctx_last_stay_days: list[int]
    ctx_same_country_streak: list[int]
    ctx_last_hotel_country: list[int]
    ctx_unique_hotel_countries: list[int]
    ctx_cross_border_count: list[int]
    ctx_cross_border_ratio: list[int]
    ctx_sem_code1: list[int]
    ctx_sem_code2: list[int]


def build_city_vocab(train_set: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """
    建立城市词表
    city_to_idx: city_id->token id
    idx_to_city: 反向映射，用于预测后还原城市
    特殊token为PAD_TOKEN_ID=0, UNK_TOKEN_ID=1
    vocab_size = len(city_to_idx)+2
    """
    unique_cities = sorted(train_set["city_id"].unique().tolist())
    city_to_idx = {city_id: idx + 2 for idx, city_id in enumerate(unique_cities)}
    idx_to_city = {idx: city for city, idx in city_to_idx.items()}
    return city_to_idx, idx_to_city


def build_city_sequence_pack(
    trip_df: pd.DataFrame,
    city_to_idx: dict[int, int],
    *,
    is_test: bool,
    multi_step: bool,
    booker_to_idx: dict[str, int],
    device_to_idx: dict[str, int],
    affiliate_to_idx: dict[str, int],
    hotel_country_to_idx: dict[str, int],
    city_to_semantic_codes: dict[int, tuple[int, int]] | None = None,
) -> CitySequencePack:
    """
    将每个trip变成模型输入
    x: 历史城市前缀token序列
    y: 下一个城市token（训练时）
    如果开启multi_step，每个trip产生多条训练样本；否则每个trip只会产生一个1样本
    """
    x_values: list[list[int]] = []
    y_values: list[int] | None = [] if not is_test else None
    cb: list[int] = []
    cd: list[int] = []
    ca: list[int] = []
    cm: list[int] = []
    cs: list[int] = []
    ctl: list[int] = []
    cnu: list[int] = []
    crr: list[int] = []
    cls: list[int] = []
    csc: list[int] = []
    clc: list[int] = []
    cuc: list[int] = []
    cbc: list[int] = []
    cbr: list[int] = []
    csem1: list[int] = []
    csem2: list[int] = []

    def append_context_and_semantic(row: pd.Series, prefix_len: int, last_city: int | None) -> None:
        b, d, a, m, s, tl, nu, rr, ls, sc = row_to_context_indices(
            row, booker_to_idx, device_to_idx, affiliate_to_idx, prefix_len=prefix_len
        )
        lc, uc, bc, br = row_to_spatial_indices(
            row, hotel_country_to_idx, prefix_len=prefix_len
        )
        sem1, sem2 = _semantic_code_from_city(last_city, city_to_semantic_codes)
        cb.append(b)
        cd.append(d)
        ca.append(a)
        cm.append(m)
        cs.append(s)
        ctl.append(tl)
        cnu.append(nu)
        crr.append(rr)
        cls.append(ls)
        csc.append(sc)
        clc.append(lc)
        cuc.append(uc)
        cbc.append(bc)
        cbr.append(br)
        csem1.append(sem1)
        csem2.append(sem2)

    for _, row in trip_df.iterrows():
        cities = row["city_id"]
        token_seq = [city_to_idx.get(city_id, UNK_TOKEN_ID) for city_id in cities]

        if is_test:
            x_values.append(token_seq[:-1])
            prefix_len = max(1, len(token_seq) - 1)
            last_city = cities[prefix_len - 1] if len(cities) >= prefix_len else None
            append_context_and_semantic(row, prefix_len, last_city)
        elif y_values is not None:
            if multi_step:
                for t in range(1, len(token_seq)):
                    x_values.append(token_seq[:t])
                    y_values.append(token_seq[t])
                    append_context_and_semantic(row, t, cities[t - 1])
            else:
                if len(token_seq) >= 2:
                    x_values.append(token_seq[:-1])
                    y_values.append(token_seq[-1])
                    prefix_len = len(token_seq) - 1
                    append_context_and_semantic(row, prefix_len, cities[prefix_len - 1])

    return CitySequencePack(
        x=x_values,
        y=y_values,
        ctx_booker=cb,
        ctx_device=cd,
        ctx_affiliate=ca,
        ctx_month=cm,
        ctx_stay=cs,
        ctx_trip_len=ctl,
        ctx_num_unique_cities=cnu,
        ctx_repeat_city_ratio=crr,
        ctx_last_stay_days=cls,
        ctx_same_country_streak=csc,
        ctx_last_hotel_country=clc,
        ctx_unique_hotel_countries=cuc,
        ctx_cross_border_count=cbc,
        ctx_cross_border_ratio=cbr,
        ctx_sem_code1=csem1,
        ctx_sem_code2=csem2,
    )

"""Word2Vec, RQ codebook, and code-sequence samples for RQ-KMeans / RQVAE."""

from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler, normalize

from src.features.context import row_to_context_indices, row_to_spatial_indices


def train_word2vec(
    train_trips: pd.DataFrame,
    vector_size: int = 128,
    window: int = 10,
) -> Word2Vec:
    """Train city-id Word2Vec on trip sequences."""
    all_city_sentences = train_trips["city_id"].apply(lambda x: [str(c) for c in x]).tolist()
    model = Word2Vec(
        all_city_sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=4,
        epochs=30,
        sg=1,
        hs=0,
    )
    return model


def _city_to_code_sequence(cities: list[int], mapping: dict[int, tuple[int, int]]) -> list[int]:
    code_seq: list[int] = []
    for city_id in cities:
        if city_id in mapping:
            code_seq.extend(list(mapping[city_id]))
        else:
            code_seq.extend([0, 0])
    return code_seq


def build_rq_codebook(
    train_set: pd.DataFrame,
    w2v: Word2Vec,
    n_clusters: int = 128,
    random_state: int = 42,
) -> dict[int, tuple[int, int]]:
    all_unique_cities = [str(c) for c in train_set["city_id"].unique()]
    raw_vectors = np.array([w2v.wv[c] for c in all_unique_cities])
    city_vectors = normalize(raw_vectors, axis=1)

    kmeans1 = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10, batch_size=2048
    )
    codes1 = kmeans1.fit_predict(city_vectors)

    residuals = city_vectors - kmeans1.cluster_centers_[codes1]
    scaler = RobustScaler()
    residuals_scaled = scaler.fit_transform(residuals)
    residuals_final = normalize(residuals_scaled, axis=1)

    kmeans2 = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10, batch_size=2048
    )
    codes2 = kmeans2.fit_predict(residuals_final)

    return {
        int(all_unique_cities[i]): (int(codes1[i]), int(codes2[i]))
        for i in range(len(all_unique_cities))
    }


def build_final_dataset(
    trip_df: pd.DataFrame,
    mapping: dict[int, tuple[int, int]],
    is_test: bool = False,
    multi_step: bool = False,
) -> tuple[list[list[int]], list[list[int]]]:
    """Build (code-prefix → next code-pair) samples for RQ-KMeans / RQVAE.

    Each city is two code tokens. **Default** (``multi_step=False``): one training row
    per trip, predicting only the **last** city's code pair. **``multi_step=True``**:
    same idea as ``build_city_sequence_pack(..., multi_step=True)`` — for a trip with
    cities ``A,B,C`` emit ``(codes(A)→codes(B))`` and ``(codes(A,B)→codes(C))``.
    Test rows (``is_test=True``) always use a single prefix per trip (all codes except
    the last pair), regardless of ``multi_step``.
    """
    x_values: list[list[int]] = []
    y_values: list[list[int]] = []

    for _, row in trip_df.iterrows():
        cities = row["city_id"]
        full_code_seq = _city_to_code_sequence(cities, mapping)

        if is_test:
            x_values.append(full_code_seq[:-2])
        elif multi_step:
            m = len(cities)
            if m < 2:
                continue
            for k in range(1, m):
                x_values.append(full_code_seq[: 2 * k])
                y_values.append(full_code_seq[2 * k : 2 * k + 2])
        else:
            if len(full_code_seq) >= 4:
                x_values.append(full_code_seq[:-2])
                y_values.append(full_code_seq[-2:])

    return x_values, y_values


def build_final_dataset_with_context(
    trip_df: pd.DataFrame,
    mapping: dict[int, tuple[int, int]],
    *,
    booker_to_idx: dict[str, int],
    device_to_idx: dict[str, int],
    hotel_country_to_idx: dict[str, int],
    is_test: bool = False,
    multi_step: bool = False,
) -> tuple[
    list[list[int]],
    list[list[int]],
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
]:
    """Build code-sequence samples and aligned trip context features."""
    x_values: list[list[int]] = []
    y_values: list[list[int]] = []
    cb: list[int] = []
    cd: list[int] = []
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

    for _, row in trip_df.iterrows():
        cities = row["city_id"]
        full_code_seq = _city_to_code_sequence(cities, mapping)
        if is_test:
            prefix_len = max(1, len(cities) - 1)
            b, d, m, s, tl, nu, rr, ls, sc = row_to_context_indices(
                row, booker_to_idx, device_to_idx, prefix_len=prefix_len
            )
            lc, uc, bc, br = row_to_spatial_indices(
                row, hotel_country_to_idx, prefix_len=prefix_len
            )
            x_values.append(full_code_seq[:-2])
            cb.append(b)
            cd.append(d)
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
        elif multi_step:
            n = len(cities)
            if n < 2:
                continue
            for k in range(1, n):
                b, d, m, s, tl, nu, rr, ls, sc = row_to_context_indices(
                    row, booker_to_idx, device_to_idx, prefix_len=k
                )
                lc, uc, bc, br = row_to_spatial_indices(
                    row, hotel_country_to_idx, prefix_len=k
                )
                x_values.append(full_code_seq[: 2 * k])
                y_values.append(full_code_seq[2 * k : 2 * k + 2])
                cb.append(b)
                cd.append(d)
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
        else:
            if len(full_code_seq) >= 4:
                prefix_len = len(cities) - 1
                b, d, m, s, tl, nu, rr, ls, sc = row_to_context_indices(
                    row, booker_to_idx, device_to_idx, prefix_len=prefix_len
                )
                lc, uc, bc, br = row_to_spatial_indices(
                    row, hotel_country_to_idx, prefix_len=prefix_len
                )
                x_values.append(full_code_seq[:-2])
                y_values.append(full_code_seq[-2:])
                cb.append(b)
                cd.append(d)
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

    return x_values, y_values, cb, cd, cm, cs, ctl, cnu, crr, cls, csc, clc, cuc, cbc, cbr


def build_code_to_cities(
    city_to_codes: dict[int, tuple[int, int]], train_set: pd.DataFrame
) -> dict[tuple[int, int], list[int]]:
    code_to_cities: dict[tuple[int, int], list[int]] = {}
    for city_id, codes in city_to_codes.items():
        code_to_cities.setdefault(tuple(codes), []).append(city_id)

    city_counts = train_set["city_id"].value_counts().to_dict()
    for code_pair in code_to_cities:
        code_to_cities[code_pair].sort(key=lambda x: city_counts.get(x, 0), reverse=True)

    return code_to_cities

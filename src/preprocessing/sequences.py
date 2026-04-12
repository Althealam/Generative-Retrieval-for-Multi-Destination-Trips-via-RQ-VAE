import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler, normalize


def create_trip_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Sort records by check-in time and aggregate rows into trip sequences."""
    data = df.copy()
    data["checkin"] = pd.to_datetime(data["checkin"])
    data = data.sort_values(["utrip_id", "checkin"])

    sequences = (
        data.groupby("utrip_id")
        .agg({"city_id": list, "hotel_country": "last", "booker_country": "first"})
        .reset_index()
    )
    return sequences


def create_mutliple_sequences(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["checkin"] = pd.to_datetime(data["checkin"])
    data["checkout"] = pd.to_datetime(data["checkout"])

    data["stay_duration"] = (data["checkout"] - data["checkin"]).dt.days
    data["stay_duration"] = data["stay_duration"].clip(1, 30)
    data["checkin_month"] = data["checkin"].dt.month

    sequences = data.groupby("utrip_id").agg(
        {
            "city_id": list,
            "stay_duration": list,
            "checkin_month": "first",
            "booker_country": "first",
            "device_class": "first",
        }
    ).reset_index()

    return sequences


def train_word2vec(train_trips, vector_size=128, window=10):
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
        full_code_seq: list[int] = []

        for city_id in cities:
            if city_id in mapping:
                full_code_seq.extend(list(mapping[city_id]))
            else:
                full_code_seq.extend([0, 0])

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

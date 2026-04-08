import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


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
    df['checkin'] = pd.to_datetime(data['checkin'])
    df['checkout'] = pd.to_datetime(data['checkout'])

    # get features
    df['stay_duration'] = (df['checkout']-df['checkin']).dt.days
    # clip duration to the range 1-30
    df['stay_duration'] = df['stay_duration'].clip(1, 30)

    df['checkin_month'] = df['checkin'].dt.month

    sequences = df.groupby('utrip_id').agg({
        'city_id': list,
        'stay_duration': list,
        'checkin_month': 'first', # only get the first month in the trip
        'booker_country': 'first', # only get the first booker country in the trip
        'device_class': 'first' # only get the first device class in the trip
    }).reset_index()

    return sequences


def train_word2vec(train_trips: pd.DataFrame, vector_size: int = 64, window: int = 5) -> Word2Vec:
    """Train Word2Vec embeddings on city sequences."""
    all_city_sentences = train_trips["city_id"].apply(lambda x: [str(c) for c in x]).tolist()
    return Word2Vec(all_city_sentences, vector_size=vector_size, window=window, min_count=1, workers=4)


def build_rq_codebook(
    train_set: pd.DataFrame,
    w2v: Word2Vec,
    n_clusters: int = 32,
    random_state: int = 42,
) -> dict[int, tuple[int, int]]:
    """Build two-level residual quantization mapping: city_id -> (code1, code2)."""
    all_unique_cities = [str(c) for c in train_set["city_id"].unique()]
    city_vectors = np.array([w2v.wv[c] for c in all_unique_cities])

    kmeans1 = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    codes1 = kmeans1.fit_predict(city_vectors)

    residuals = city_vectors - kmeans1.cluster_centers_[codes1]

    kmeans2 = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    codes2 = kmeans2.fit_predict(residuals)

    city_to_codes = {int(all_unique_cities[i]): (int(codes1[i]), int(codes2[i])) for i in range(len(all_unique_cities))}
    return city_to_codes


def build_final_dataset(
    trip_df: pd.DataFrame, mapping: dict[int, tuple[int, int]], is_test: bool = False
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Convert city sequences into code sequences.

    - Train: X uses first N-1 stations, y uses last station code pair.
    - Test: X removes the final placeholder station.
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
        else:
            if len(full_code_seq) >= 4:
                x_values.append(full_code_seq[:-2])
                y_values.append(full_code_seq[-2:])

    return x_values, y_values


def build_code_to_cities(
    city_to_codes: dict[int, tuple[int, int]], train_set: pd.DataFrame
) -> dict[tuple[int, int], list[int]]:
    """Build reverse index (code1, code2) -> city_ids sorted by popularity."""
    code_to_cities: dict[tuple[int, int], list[int]] = {}
    for city_id, codes in city_to_codes.items():
        code_to_cities.setdefault(tuple(codes), []).append(city_id)

    city_counts = train_set["city_id"].value_counts().to_dict()
    for code_pair in code_to_cities:
        code_to_cities[code_pair].sort(key=lambda x: city_counts.get(x, 0), reverse=True)

    return code_to_cities

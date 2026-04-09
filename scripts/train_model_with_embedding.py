from datetime import datetime
from pathlib import Path
import os
import pandas as pd

from src.embedding_transformer_dataset import PAD_TOKEN_ID, UNK_TOKEN_ID, build_city_dataloaders
from src.embedding_transformer_model import CityTransformer
from src.embedding_transformer_train import (
    compute_top_popular_cities,
    recommend_top4_cities,
    train_city_transformer,
)
from src.preprocessing import create_trip_sequences


def build_city_vocab(train_set: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    unique_cities = sorted(train_set["city_id"].unique().tolist())
    city_to_idx = {city_id: idx + 2 for idx, city_id in enumerate(unique_cities)}
    idx_to_city = {idx: city for city, idx in city_to_idx.items()}
    return city_to_idx, idx_to_city


def build_city_dataset(
    trip_df: pd.DataFrame,
    city_to_idx: dict[int, int],
    is_test: bool = False,
) -> tuple[list[list[int]], list[int]]:
    x_values: list[list[int]] = []
    y_values: list[int] = []

    for _, row in trip_df.iterrows():
        cities = row["city_id"]
        token_seq = [city_to_idx.get(city_id, UNK_TOKEN_ID) for city_id in cities]

        if is_test:
            x_values.append(token_seq[:-1])
        else:
            if len(token_seq) >= 2:
                x_values.append(token_seq[:-1])
                y_values.append(token_seq[-1])

    return x_values, y_values


def main():
    # root_dir = Path(__file__).resolve().parents[1]
    # data_dir = root_dir / "data"
    # output_dir = root_dir / "output"
    # output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = "/Users/althealam/Desktop/GitHub/Generative-Retrieval-for-Multi-Destination-Trips-via-RQ-VAE/output/submission"

    train_set = pd.read_csv('/Users/althealam/Desktop/GitHub/Generative-Retrieval-for-Multi-Destination-Trips-via-RQ-VAE/data/train_set.csv')
    test_set = pd.read_csv('/Users/althealam/Desktop/GitHub/Generative-Retrieval-for-Multi-Destination-Trips-via-RQ-VAE/data/test_set.csv')
    # train_set = pd.read_csv(data_dir / "train_set.csv")
    # test_set = pd.read_csv(data_dir / "test_set.csv")

    print("正在聚合行程序列...")
    train_trips = create_trip_sequences(train_set)
    test_trips = create_trip_sequences(test_set)

    city_to_idx, idx_to_city = build_city_vocab(train_set)
    vocab_size = len(city_to_idx) + 2  # 0: PAD, 1: UNK

    train_x, train_y = build_city_dataset(train_trips, city_to_idx, is_test=False)
    test_x, _ = build_city_dataset(test_trips, city_to_idx, is_test=True)
    print(f"✅ City 数据集完成！训练样本: {len(train_x)} | 测试样本: {len(test_x)}")

    train_loader, test_loader = build_city_dataloaders(train_x, train_y, test_x, batch_size=256)
    model = CityTransformer(vocab_size=vocab_size, pad_token_id=PAD_TOKEN_ID, d_model=256, nhead=4, num_layers=2)
    model = train_city_transformer(model, train_loader, pad_token_id=PAD_TOKEN_ID, epochs=5, lr=1e-3)

    top_popular = compute_top_popular_cities(train_set["city_id"], k=4)
    predictions = recommend_top4_cities(
        model=model,
        test_loader=test_loader,
        idx_to_city=idx_to_city,
        top_popular_cities=top_popular,
        reserved_token_ids={PAD_TOKEN_ID, UNK_TOKEN_ID},
        topk_candidates=50,
    )

    submission_df = pd.DataFrame(
        predictions,
        columns=["city_id_1", "city_id_2", "city_id_3", "city_id_4"],
    )
    submission_df.insert(0, "utrip_id", test_trips["utrip_id"].tolist())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = os.path.join(output_dir, f"submission_embedding_{timestamp}.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"✅ {submission_path} 已生成！")


if __name__ == "__main__":
    main()

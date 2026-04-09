import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import build_dataloaders
from src.preprocessing import build_code_to_cities, build_final_dataset, create_mutliple_sequences
from src.rqvae_train_infer import predict_top4_cities_from_true_rqvae, train_true_rqvae_model
from src.rqvae_transformer import TrueRQVAETransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_path", type=str, default=None, help="Path to city_to_codes_true_rqvae json.")
    parser.add_argument("--codebook_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def auto_find_latest_mapping(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("city_to_codes_rqvae_*.json"))
    if not candidates:
        raise FileNotFoundError(
            "No city_to_codes_true_rqvae_*.json found. Run scripts.train_true_rqvae_codebook first."
        )
    return candidates[-1]


def load_mapping(path: Path) -> dict[int, tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): (int(v[0]), int(v[1])) for k, v in raw.items()}


def main():
    args = parse_args()

    # root_dir = Path(__file__).resolve().parents[1]
    # data_dir = root_dir / "data"
    output_dir = "/Users/althealam/Desktop/GitHub/Generative-Retrieval-for-Multi-Destination-Trips-via-RQ-VAE/output/submission"
    # output_dir = root_dir / "output"
    # output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = Path(args.mapping_path) if args.mapping_path else auto_find_latest_mapping(output_dir)
    city_to_codes = load_mapping(mapping_path)
    print(f"Using mapping: {mapping_path}")

    train_set = pd.read_csv('/Users/althealam/Desktop/GitHub/Generative-Retrieval-for-Multi-Destination-Trips-via-RQ-VAE/data/train_set.csv')
    test_set = pd.read_csv('/Users/althealam/Desktop/GitHub/Generative-Retrieval-for-Multi-Destination-Trips-via-RQ-VAE/data/test_set.csv')
    # train_set = pd.read_csv(data_dir / "train_set.csv")
    # test_set = pd.read_csv(data_dir / "test_set.csv")

    print("正在聚合行程序列...")
    train_trips = create_mutliple_sequences(train_set)
    test_trips = create_mutliple_sequences(test_set)

    train_x, train_y = build_final_dataset(train_trips, city_to_codes, is_test=False)
    test_x, _ = build_final_dataset(test_trips, city_to_codes, is_test=True)
    print(f"✅ 数据集构建完成！训练样本: {len(train_x)} | 测试样本: {len(test_x)}")

    train_loader, test_loader = build_dataloaders(train_x, train_y, test_x, batch_size=args.batch_size)
    model = TrueRQVAETransformer(
        codebook_size=args.codebook_size,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
    )
    model = train_true_rqvae_model(model, train_loader, epochs=args.epochs, lr=args.lr)

    code_to_cities = build_code_to_cities(city_to_codes, train_set)
    predictions = predict_top4_cities_from_true_rqvae(
        model=model,
        test_loader=test_loader,
        code_to_cities=code_to_cities,
        codebook_size=args.codebook_size,
        topk_pairs=100,
    )

    submission_df = pd.DataFrame(
        predictions,
        columns=["city_id_1", "city_id_2", "city_id_3", "city_id_4"],
    )
    submission_df.insert(0, "utrip_id", test_trips["utrip_id"].tolist())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = os.path.join(output_dir, f"submission_rqvae_transformer_{timestamp}.csv")
    # submission_path = output_dir / f"submission_true_rqvae_transformer_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"✅ {submission_path} 已生成！")


if __name__ == "__main__":
    main()

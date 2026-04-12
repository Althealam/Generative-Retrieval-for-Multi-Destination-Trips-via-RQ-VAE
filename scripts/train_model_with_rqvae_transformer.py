import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import build_dataloaders
from src.models import RQVAETransformer
from src.preprocessing import build_code_to_cities, build_final_dataset, create_mutliple_sequences
from src.training.rqvae import predict_top4_cities_from_rqvae, train_rqvae_model
from src.utils import data_dir, print_accuracy_at_4_report, rqvae_dir, submission_dir, top_city_ids_from_train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_path", type=str, default=None, help="Path to city_to_codes_rqvae json.")
    parser.add_argument("--codebook_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip_eval", action="store_true", help="Do not run Accuracy@4 vs data/ground_truth.csv after training.")
    parser.add_argument("--ground_truth", type=str, default=None, help="Optional path to ground_truth.csv (default: data/ground_truth.csv).")
    parser.add_argument(
        "--multi_step",
        action="store_true",
        help="Expand each trip into many prefix→next-code-pair training rows (test inference unchanged).",
    )
    return parser.parse_args()


def auto_find_latest_mapping(rqvae_out: Path) -> Path:
    candidates = sorted(rqvae_out.glob("city_to_codes_rqvae_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No city_to_codes_rqvae_*.json under {rqvae_out}. "
            "Run: python -m scripts.train_rqvae_codebook"
        )
    return candidates[-1]


def load_mapping(path: Path) -> dict[int, tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): (int(v[0]), int(v[1])) for k, v in raw.items()}


def main():
    args = parse_args()

    rqvae_out = rqvae_dir()
    sub_dir = submission_dir()
    sub_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = Path(args.mapping_path) if args.mapping_path else auto_find_latest_mapping(rqvae_out)
    city_to_codes = load_mapping(mapping_path)
    print(f"Using mapping: {mapping_path}")

    train_set = pd.read_csv(data_dir() / "train_set.csv")
    test_set = pd.read_csv(data_dir() / "test_set.csv")

    print("正在聚合行程序列...")
    train_trips = create_mutliple_sequences(train_set)
    test_trips = create_mutliple_sequences(test_set)

    train_x, train_y = build_final_dataset(
        train_trips, city_to_codes, is_test=False, multi_step=args.multi_step
    )
    test_x, _ = build_final_dataset(test_trips, city_to_codes, is_test=True)
    print(
        f"✅ 数据集构建完成！训练样本: {len(train_x)} | 测试样本: {len(test_x)} | multi_step={args.multi_step}"
    )

    train_loader, test_loader = build_dataloaders(
        train_x,
        train_y,
        test_x,
        batch_size=args.batch_size,
        pad_token=args.codebook_size,
    )
    model = RQVAETransformer(
        codebook_size=args.codebook_size,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
    )
    model = train_rqvae_model(model, train_loader, epochs=args.epochs, lr=args.lr)

    code_to_cities = build_code_to_cities(city_to_codes, train_set)
    fallback_cities = top_city_ids_from_train(train_set, k=4)
    predictions = predict_top4_cities_from_rqvae(
        model=model,
        test_loader=test_loader,
        code_to_cities=code_to_cities,
        codebook_size=args.codebook_size,
        top_global=fallback_cities,
        topk_pairs=100,
    )

    submission_df = pd.DataFrame(
        predictions,
        columns=["city_id_1", "city_id_2", "city_id_3", "city_id_4"],
    )
    submission_df.insert(0, "utrip_id", test_trips["utrip_id"].tolist())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = sub_dir / f"submission_rqvae_transformer_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"✅ {submission_path} 已生成！")
    print_accuracy_at_4_report(
        submission_df,
        skip=args.skip_eval,
        ground_truth_path=args.ground_truth,
    )


if __name__ == "__main__":
    main()

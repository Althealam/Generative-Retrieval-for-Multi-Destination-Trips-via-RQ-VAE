"""RQVAE-derived city code mapping + Transformer (loads JSON mapping from train_rqvae_codebook)."""

from __future__ import annotations

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
from src.features import (
    build_booker_device_affiliate_vocabs,
    build_hotel_country_vocab,
    build_code_to_cities,
    build_final_dataset_with_context,
    create_multiple_sequences,
)
from src.models import RQVAETransformer, RQVAEGRU
from src.training.code_predict import predict_top4_with_codebook, train_code_transformer
from src.utils import data_dir, print_accuracy_at_4_report, rqvae_dir, submission_dir, top_city_ids_from_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping_path", type=str, default=None, help="city_to_codes JSON (default: latest under output/rqvae).")
    p.add_argument("--codebook_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--skip_eval", action="store_true", help="Skip Accuracy@4 vs ground truth.")
    p.add_argument("--ground_truth", type=str, default=None, help="Path to ground_truth.csv.")
    p.add_argument(
        "--multi_step",
        action="store_true",
        help="Many prefix→next-code-pair rows per trip for training (test unchanged).",
    )
    p.add_argument("--model", type=str, default='transformer', help="choose the prediction model type: transformer, gru, lstm...")
    return p.parse_args()


def auto_find_latest_mapping(rqvae_out: Path) -> Path:
    candidates = sorted(rqvae_out.glob("city_to_codes_rqvae_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No city_to_codes_rqvae_*.json under {rqvae_out}. Run: python -m scripts.train_rqvae_codebook"
        )
    return candidates[-1]


def load_mapping(path: Path) -> dict[int, tuple[int, int]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): (int(v[0]), int(v[1])) for k, v in raw.items()}


def main() -> None:
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
    train_trips = create_multiple_sequences(train_set)
    test_trips = create_multiple_sequences(test_set)

    booker_to_idx, device_to_idx, affiliate_to_idx, n_booker, n_device, n_affiliate = (
        build_booker_device_affiliate_vocabs(train_trips)
    )
    hotel_country_to_idx, n_hotel_country = build_hotel_country_vocab(train_trips)

    train_parts = build_final_dataset_with_context(
        train_trips,
        city_to_codes,
        booker_to_idx=booker_to_idx,
        device_to_idx=device_to_idx,
        affiliate_to_idx=affiliate_to_idx,
        hotel_country_to_idx=hotel_country_to_idx,
        is_test=False,
        multi_step=args.multi_step,
    )
    train_x, train_y, *train_ctx = train_parts

    test_parts = build_final_dataset_with_context(
        test_trips,
        city_to_codes,
        booker_to_idx=booker_to_idx,
        device_to_idx=device_to_idx,
        affiliate_to_idx=affiliate_to_idx,
        hotel_country_to_idx=hotel_country_to_idx,
        is_test=True,
    )
    test_x, _, *test_ctx = test_parts

    print(
        f"✅ 数据集构建完成！训练样本: {len(train_x)} | 测试样本: {len(test_x)} | multi_step={args.multi_step} | model_type={args.model}"
    )

    train_loader, test_loader = build_dataloaders(
        train_x,
        train_y,
        test_x,
        batch_size=args.batch_size,
        pad_token=args.codebook_size,
        train_ctx=tuple(train_ctx),
        test_ctx=tuple(test_ctx),
    )

    model_type = args.model
    if model_type == 'transformer':
        model = RQVAETransformer(
            codebook_size=args.codebook_size,
            d_model=256,
            nhead=4,
            num_layers=2,
            dim_feedforward=512,
            n_booker_countries=n_booker,
            n_device_classes=n_device,
            n_affiliates=n_affiliate,
            n_hotel_countries=n_hotel_country,
        )
    elif model_type == "gru":
        model = RQVAEGRU(
            codebook_size=args.codebook_size,
            embedding_dim=64,
            hidden_dim=128,
            n_booker_countries=n_booker,
            n_device_classes=n_device,
            n_affiliates=n_affiliate,
            n_hotel_countries=n_hotel_country,
        )
    else:
        raise ValueError(f"Unknown --model {model_type!r}; use 'transformer' or 'gru'.")
    model = train_code_transformer(model, train_loader, epochs=args.epochs, lr=args.lr)

    code_to_cities = build_code_to_cities(city_to_codes, train_set)
    fallback_cities = top_city_ids_from_train(train_set, k=4)
    predictions = predict_top4_with_codebook(
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
    submission_path = sub_dir / f"submission_rqvae_{model_type}_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"✅ {submission_path} 已生成！")
    print_accuracy_at_4_report(
        submission_df,
        skip=args.skip_eval,
        ground_truth_path=args.ground_truth,
    )


if __name__ == "__main__":
    main()

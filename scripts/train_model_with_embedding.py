"""Train direct city-id Transformer with trip context embeddings."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import PAD_TOKEN_ID, UNK_TOKEN_ID, build_city_dataloaders
from src.features import (
    build_booker_device_vocabs,
    build_city_sequence_pack,
    build_city_vocab,
    create_multiple_sequences,
)
from src.models import CityGRU, CityTransformer
from src.training.embedding import recommend_top4_cities, train_embedding_model
from src.utils import data_dir, print_accuracy_at_4_report, submission_dir, top_city_ids_from_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument(
        "--multi_step",
        action="store_true",
        help="Expand each trip into many prefix→next-city samples (larger / slower).",
    )
    p.add_argument("--topk_candidates", type=int, default=50)
    p.add_argument("--skip_eval", action="store_true", help="Skip Accuracy@4 vs ground truth.")
    p.add_argument("--ground_truth", type=str, default=None, help="Path to ground_truth.csv (default: data/).")
    p.add_argument("--model", type=str, default='transformer', help="transformer/gru")
    p.add_argument("--pooling", type=str, default="last", help="last/mean/cls (Transformer only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = submission_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = pd.read_csv(data_dir() / "train_set.csv")
    test_set = pd.read_csv(data_dir() / "test_set.csv")

    print("正在聚合行程序列...")
    train_trips = create_multiple_sequences(train_set)
    # print(train_trips)
    test_trips = create_multiple_sequences(test_set)

    city_to_idx, idx_to_city = build_city_vocab(train_set)
    # print("city_to_idx:", city_to_idx)
    vocab_size = len(city_to_idx) + 2

    booker_to_idx, device_to_idx, n_booker, n_device = build_booker_device_vocabs(train_trips)

    train_pack = build_city_sequence_pack(
        train_trips,
        city_to_idx,
        is_test=False,
        multi_step=args.multi_step,
        booker_to_idx=booker_to_idx,
        device_to_idx=device_to_idx,
    )
    test_pack = build_city_sequence_pack(
        test_trips,
        city_to_idx,
        is_test=True,
        multi_step=False,
        booker_to_idx=booker_to_idx,
        device_to_idx=device_to_idx,
    )

    print(
        f"✅ City 数据集完成！训练样本: {len(train_pack.x)} | 测试样本: {len(test_pack.x)} "
        f"| multi_step={args.multi_step}"
        f"| model={args.model}"
        f"| pooling={args.pooling}"
    )

    train_ctx = (
        train_pack.ctx_booker,
        train_pack.ctx_device,
        train_pack.ctx_month,
        train_pack.ctx_stay,
        train_pack.ctx_trip_len,
        train_pack.ctx_num_unique_cities,
        train_pack.ctx_repeat_city_ratio,
        train_pack.ctx_last_stay_days,
        train_pack.ctx_same_country_streak,
    )
    test_ctx = (
        test_pack.ctx_booker,
        test_pack.ctx_device,
        test_pack.ctx_month,
        test_pack.ctx_stay,
        test_pack.ctx_trip_len,
        test_pack.ctx_num_unique_cities,
        test_pack.ctx_repeat_city_ratio,
        test_pack.ctx_last_stay_days,
        test_pack.ctx_same_country_streak,
    )
    train_loader, test_loader = build_city_dataloaders(
        train_pack.x,
        train_pack.y,
        test_pack.x,
        batch_size=args.batch_size,
        train_ctx=train_ctx,
        test_ctx=test_ctx,
    )
    
    model_type = args.model
    if model_type=='transformer':
        model = CityTransformer(
            vocab_size=vocab_size,
            pad_token_id=PAD_TOKEN_ID,
            d_model=256,
            nhead=4,
            num_layers=2,
            n_booker_countries=n_booker,
            n_device_classes=n_device,
            pooling=args.pooling,
        )
        model = train_embedding_model(
            model,
            train_loader,
            pad_token_id=PAD_TOKEN_ID,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
        )
    elif model_type == "gru":
        model = CityGRU(
            vocab_size=vocab_size,
            pad_token_id=PAD_TOKEN_ID,
            embedding_dim=256,
            hidden_dim=256,
            n_booker_countries=n_booker,
            n_device_classes=n_device,
        )
        model = train_embedding_model(
            model,
            train_loader,
            pad_token_id=PAD_TOKEN_ID,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    top_popular = top_city_ids_from_train(train_set, k=4)
    predictions = recommend_top4_cities(
        model=model,
        test_loader=test_loader,
        idx_to_city=idx_to_city,
        top_popular_cities=top_popular,
        reserved_token_ids={PAD_TOKEN_ID, UNK_TOKEN_ID},
        topk_candidates=args.topk_candidates,
    )

    submission_df = pd.DataFrame(
        predictions,
        columns=["city_id_1", "city_id_2", "city_id_3", "city_id_4"],
    )
    submission_df.insert(0, "utrip_id", test_trips["utrip_id"].tolist())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = out_dir / f"submission_embedding_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"✅ {submission_path} 已生成！")
    print_accuracy_at_4_report(
        submission_df,
        skip=args.skip_eval,
        ground_truth_path=args.ground_truth,
    )


if __name__ == "__main__":
    main()

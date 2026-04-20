"""Train direct city-id Transformer with trip context embeddings."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import PAD_TOKEN_ID, UNK_TOKEN_ID, build_city_dataloaders
from src.features import (
    build_booker_device_affiliate_vocabs,
    build_hotel_country_vocab,
    build_rq_codebook,
    build_city_sequence_pack,
    build_city_vocab,
    create_multiple_sequences,
    train_word2vec,
)
from src.models import CityGRU, CityTransformer
from src.training.embedding import recommend_top4_cities, train_embedding_model
from src.utils import data_dir, print_accuracy_at_4_report, submission_dir, top_city_ids_from_train


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
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
    p.add_argument(
        "--fusion",
        type=str,
        default="add",
        choices=["add", "gate"],
        help="How to fuse context with sequence representation.",
    )
    p.add_argument(
        "--semantic_source",
        type=str,
        default="none",
        choices=["none", "rqkmeans", "rqvae"],
        help="Optional semantic side info source.",
    )
    p.add_argument(
        "--semantic_mapping_path",
        type=str,
        default="/root/gr/Generative-Retrieval-for-Multi-Destination-Trips/output/rqvae/city_to_codes_rqvae_20260409_110222.json",
        help="Path to city_to_codes JSON (required for --semantic_source rqvae).",
    )
    p.add_argument(
        "--semantic_codebook_size",
        type=int,
        default=128,
        help="Semantic codebook size (e.g., 32 for rqkmeans, 128 for rqvae).",
    )
    return p.parse_args()

def _load_rqvae_mapping(mapping_path: str) -> dict[int, tuple[int, int]]:
    with open(mapping_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): (int(v[0]), int(v[1])) for k, v in raw.items()}


def _build_semantic_mapping(
    args: argparse.Namespace,
    train_set: pd.DataFrame,
    train_trips: pd.DataFrame,
) -> dict[int, tuple[int, int]] | None:
    if args.semantic_source == "none":
        return None
    if args.semantic_source == "rqkmeans":
        print("正在构建 RQ-KMeans semantic id 映射...")
        w2v = train_word2vec(train_trips, vector_size=64, window=5)
        return build_rq_codebook(
            train_set,
            w2v,
            n_clusters=args.semantic_codebook_size,
            random_state=args.seed,
        )
    if not args.semantic_mapping_path:
        raise ValueError("--semantic_mapping_path is required when --semantic_source rqvae")
    print("正在加载 RQVAE semantic id 映射...")
    return _load_rqvae_mapping(args.semantic_mapping_path)


def _ctx_tuple_from_pack(pack) -> tuple[list[int], ...]:
    return (
        pack.ctx_booker,
        pack.ctx_device,
        pack.ctx_affiliate,
        pack.ctx_month,
        pack.ctx_stay,
        pack.ctx_trip_len,
        pack.ctx_num_unique_cities,
        pack.ctx_repeat_city_ratio,
        pack.ctx_last_stay_days,
        pack.ctx_same_country_streak,
        pack.ctx_last_hotel_country,
        pack.ctx_unique_hotel_countries,
        pack.ctx_cross_border_count,
        pack.ctx_cross_border_ratio,
        pack.ctx_sem_code1,
        pack.ctx_sem_code2,
    )


def _build_model(
    args: argparse.Namespace,
    *,
    vocab_size: int,
    n_booker: int,
    n_device: int,
    n_affiliate: int,
    n_hotel_country: int,
):
    n_semantic_codes = args.semantic_codebook_size if args.semantic_source != "none" else 0
    if args.model == "transformer":
        return CityTransformer(
            vocab_size=vocab_size,
            pad_token_id=PAD_TOKEN_ID,
            d_model=256,
            nhead=4,
            num_layers=2,
            n_booker_countries=n_booker,
            n_device_classes=n_device,
            n_affiliates=n_affiliate,
            n_hotel_countries=n_hotel_country,
            n_semantic_codes=n_semantic_codes,
            pooling=args.pooling,
            fusion=args.fusion,
        )
    if args.model == "gru":
        return CityGRU(
            vocab_size=vocab_size,
            pad_token_id=PAD_TOKEN_ID,
            embedding_dim=256,
            hidden_dim=256,
            n_booker_countries=n_booker,
            n_device_classes=n_device,
            n_affiliates=n_affiliate,
            n_hotel_countries=n_hotel_country,
            n_semantic_codes=n_semantic_codes,
            fusion=args.fusion,
        )
    raise ValueError(f"Unsupported model type: {args.model}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    print(f"Using random seed: {args.seed}")
    out_dir = submission_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = pd.read_csv(data_dir() / "train_set.csv")
    test_set = pd.read_csv(data_dir() / "test_set.csv")

    print("正在聚合行程序列...")
    train_trips = create_multiple_sequences(train_set)
    # print(train_trips)
    test_trips = create_multiple_sequences(test_set)

    city_to_semantic_codes = _build_semantic_mapping(args, train_set, train_trips)

    city_to_idx, idx_to_city = build_city_vocab(train_set)
    # print("city_to_idx:", city_to_idx)
    vocab_size = len(city_to_idx) + 2

    booker_to_idx, device_to_idx, affiliate_to_idx, n_booker, n_device, n_affiliate = (
        build_booker_device_affiliate_vocabs(train_trips)
    )
    hotel_country_to_idx, n_hotel_country = build_hotel_country_vocab(train_trips)

    train_pack = build_city_sequence_pack(
        train_trips,
        city_to_idx,
        is_test=False,
        multi_step=args.multi_step,
        booker_to_idx=booker_to_idx,
        device_to_idx=device_to_idx,
        affiliate_to_idx=affiliate_to_idx,
        hotel_country_to_idx=hotel_country_to_idx,
        city_to_semantic_codes=city_to_semantic_codes,
    )
    test_pack = build_city_sequence_pack(
        test_trips,
        city_to_idx,
        is_test=True,
        multi_step=False,
        booker_to_idx=booker_to_idx,
        device_to_idx=device_to_idx,
        affiliate_to_idx=affiliate_to_idx,
        hotel_country_to_idx=hotel_country_to_idx,
        city_to_semantic_codes=city_to_semantic_codes,
    )

    print(
        f"✅ City 数据集完成！训练样本: {len(train_pack.x)} | 测试样本: {len(test_pack.x)} "
        f"| multi_step={args.multi_step}"
        f"| model={args.model}"
        f"| pooling={args.pooling}"
    )

    train_ctx = _ctx_tuple_from_pack(train_pack)
    test_ctx = _ctx_tuple_from_pack(test_pack)
    train_loader, test_loader = build_city_dataloaders(
        train_pack.x,
        train_pack.y,
        test_pack.x,
        batch_size=args.batch_size,
        train_ctx=train_ctx,
        test_ctx=test_ctx,
    )
    
    model = _build_model(
        args,
        vocab_size=vocab_size,
        n_booker=n_booker,
        n_device=n_device,
        n_affiliate=n_affiliate,
        n_hotel_country=n_hotel_country,
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

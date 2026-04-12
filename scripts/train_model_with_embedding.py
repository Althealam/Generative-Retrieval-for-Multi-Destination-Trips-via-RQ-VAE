"""Train direct city-id Transformer; optional trip context + multi-step next-city targets."""

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
from src.models import CityTransformer
from src.preprocessing import (
    build_booker_device_vocabs,
    build_city_sequence_pack,
    build_city_vocab,
    create_mutliple_sequences,
)
from src.training.embedding import recommend_top4_cities, train_city_transformer
from src.utils import data_dir, print_accuracy_at_4_report, submission_dir, top_city_ids_from_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--no_context", action="store_true", help="Disable booker/device/month/stay embeddings")
    p.add_argument(
        "--multi_step",
        action="store_true",
        default=False,
        help="If set, expand each trip into many prefix→next-city samples (much larger/slower). Default: off.",
    )
    p.add_argument("--topk_candidates", type=int, default=50)
    p.add_argument("--skip_eval", action="store_true", help="Do not run Accuracy@4 vs data/ground_truth.csv after training.")
    p.add_argument("--ground_truth", type=str, default=None, help="Optional path to ground_truth.csv (default: data/ground_truth.csv).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = submission_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = pd.read_csv(data_dir() / "train_set.csv")
    test_set = pd.read_csv(data_dir() / "test_set.csv")

    print("正在聚合行程序列...")
    train_trips = create_mutliple_sequences(train_set)
    test_trips = create_mutliple_sequences(test_set)

    city_to_idx, idx_to_city = build_city_vocab(train_set)
    vocab_size = len(city_to_idx) + 2

    use_context = not args.no_context
    booker_to_idx = device_to_idx = None
    n_booker = n_device = 0
    if use_context:
        booker_to_idx, device_to_idx, n_booker, n_device = build_booker_device_vocabs(train_trips)

    train_pack = build_city_sequence_pack(
        train_trips,
        city_to_idx,
        is_test=False,
        multi_step=args.multi_step,
        use_context=use_context,
        booker_to_idx=booker_to_idx,
        device_to_idx=device_to_idx,
    )
    test_pack = build_city_sequence_pack(
        test_trips,
        city_to_idx,
        is_test=True,
        multi_step=False,
        use_context=use_context,
        booker_to_idx=booker_to_idx,
        device_to_idx=device_to_idx,
    )

    print(
        f"✅ City 数据集完成！训练样本: {len(train_pack.x)} | 测试样本: {len(test_pack.x)} "
        f"| context={use_context} | multi_step={args.multi_step}"
    )

    if use_context:
        train_ctx = (train_pack.ctx_booker, train_pack.ctx_device, train_pack.ctx_month, train_pack.ctx_stay)
        test_ctx = (test_pack.ctx_booker, test_pack.ctx_device, test_pack.ctx_month, test_pack.ctx_stay)
        train_loader, test_loader = build_city_dataloaders(
            train_pack.x,
            train_pack.y,
            test_pack.x,
            batch_size=args.batch_size,
            train_ctx=train_ctx,
            test_ctx=test_ctx,
        )
    else:
        train_loader, test_loader = build_city_dataloaders(
            train_pack.x, train_pack.y, test_pack.x, batch_size=args.batch_size
        )

    model = CityTransformer(
        vocab_size=vocab_size,
        pad_token_id=PAD_TOKEN_ID,
        d_model=256,
        nhead=4,
        num_layers=2,
        use_trip_context=use_context,
        n_booker_countries=n_booker,
        n_device_classes=n_device,
    )
    model = train_city_transformer(
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

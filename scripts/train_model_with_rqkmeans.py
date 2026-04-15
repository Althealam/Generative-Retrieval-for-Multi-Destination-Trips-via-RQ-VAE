"""RQ-KMeans codebook + Transformer (pad token 32 when K=32, codes 0..31)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import DEFAULT_CODE_PAD_TOKEN, build_dataloaders
from src.features import (
    build_booker_device_vocabs,
    build_hotel_country_vocab,
    build_code_to_cities,
    build_final_dataset_with_context,
    build_rq_codebook,
    create_multiple_sequences,
    train_word2vec,
)
from src.models import RQKMeansTransformer
from src.models.rqkmeans.gru import RQKmeansGRU
from src.training.code_predict import predict_top4_with_codebook, train_code_transformer
from src.utils import data_dir, print_accuracy_at_4_report, submission_dir, top_city_ids_from_train

N_CLUSTERS = 32

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip_eval", action="store_true", help="Skip Accuracy@4 vs ground truth.")
    p.add_argument("--ground_truth", type=str, default=None, help="Path to ground_truth.csv.")
    p.add_argument("--model", type=str, default="transformer", help="choose the prediction model type: transformer, gru, lstm...")
    p.add_argument(
        "--multi_step",
        action="store_true",
        help="Many prefix→next-code-pair rows per trip for training (test unchanged).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = submission_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = pd.read_csv(data_dir() / "train_set.csv")
    test_set = pd.read_csv(data_dir() / "test_set.csv")

    print("正在聚合行程序列...")
    train_trips = create_multiple_sequences(train_set)
    test_trips = create_multiple_sequences(test_set)

    print("正在训练 Word2Vec...")
    w2v = train_word2vec(train_trips, vector_size=64, window=5)

    print("正在构建 RQ 语义索引 (Residual Quantization)...")
    city_to_codes = build_rq_codebook(train_set, w2v, n_clusters=N_CLUSTERS, random_state=42)
    
    booker_to_idx, device_to_idx, n_booker, n_device = build_booker_device_vocabs(train_trips)
    hotel_country_to_idx, n_hotel_country = build_hotel_country_vocab(train_trips)

    train_parts = build_final_dataset_with_context(
        train_trips,
        city_to_codes,
        booker_to_idx=booker_to_idx,
        device_to_idx=device_to_idx,
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
        hotel_country_to_idx=hotel_country_to_idx,
        is_test=True,
    )
    test_x, _, *test_ctx = test_parts

    print("✅ 数据集构建完成！")
    print(f"训练集样本: {len(train_x)} | 测试集样本: {len(test_x)} | multi_step={args.multi_step} | model_type={args.model}")

    train_loader, test_loader = build_dataloaders(
        train_x,
        train_y,
        test_x,
        batch_size=256,
        pad_token=DEFAULT_CODE_PAD_TOKEN,
        train_ctx=tuple(train_ctx),
        test_ctx=tuple(test_ctx),
    )
    model_type = args.model
    if model_type == "transformer":
        model = RQKMeansTransformer(
            n_booker_countries=n_booker,
            n_device_classes=n_device,
            n_hotel_countries=n_hotel_country,
        )
    elif model_type == "gru":
        model = RQKmeansGRU(
            n_booker_countries=n_booker,
            n_device_classes=n_device,
            n_hotel_countries=n_hotel_country,
            codebook_size=N_CLUSTERS,
        )
    else:
        raise ValueError(f"Unknown --model {model_type!r}; use 'transformer' or 'gru'.")

    model = train_code_transformer(model, train_loader, epochs=5, lr=0.001)

    code_to_cities = build_code_to_cities(city_to_codes, train_set)
    fallback = top_city_ids_from_train(train_set, k=4)
    all_predictions = predict_top4_with_codebook(
        model,
        test_loader,
        code_to_cities,
        codebook_size=N_CLUSTERS,
        top_global=fallback,
        topk_pairs=50,
    )

    submission_df = pd.DataFrame(
        all_predictions,
        columns=["city_id_1", "city_id_2", "city_id_3", "city_id_4"],
    )
    submission_df.insert(0, "utrip_id", test_trips["utrip_id"].tolist())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = out_dir / f"submission_rqkmeans_{model_type}_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"✅ {submission_path} 已生成！")
    print_accuracy_at_4_report(
        submission_df,
        skip=args.skip_eval,
        ground_truth_path=args.ground_truth,
    )


if __name__ == "__main__":
    main()

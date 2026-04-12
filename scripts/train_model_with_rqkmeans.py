"""RQ-KMeans codebook + Transformer; pad token 32 when K=32 (codes 0..31)."""

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
from src.models import RQKMeansTransformer
from src.preprocessing import (
    build_code_to_cities,
    build_final_dataset,
    build_rq_codebook,
    create_mutliple_sequences,
    train_word2vec,
)
from src.training.rqkmeans import predict_top4_cities, train_model
from src.utils import data_dir, print_accuracy_at_4_report, submission_dir, top_city_ids_from_train

N_CLUSTERS = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip_eval", action="store_true", help="Do not run Accuracy@4 vs data/ground_truth.csv after training.")
    p.add_argument("--ground_truth", type=str, default=None, help="Optional path to ground_truth.csv (default: data/ground_truth.csv).")
    p.add_argument(
        "--multi_step",
        action="store_true",
        help="Expand each trip into many prefix→next-code-pair training rows (test inference unchanged).",
    )
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

    print("正在训练 Word2Vec...")
    w2v = train_word2vec(train_trips, vector_size=64, window=5)

    print("正在构建 RQ 语义索引 (Residual Quantization)...")
    city_to_codes = build_rq_codebook(train_set, w2v, n_clusters=N_CLUSTERS, random_state=42)

    train_x, train_y = build_final_dataset(
        train_trips, city_to_codes, is_test=False, multi_step=args.multi_step
    )
    test_x, _ = build_final_dataset(test_trips, city_to_codes, is_test=True)
    print("✅ 数据集构建完成！")
    print(f"训练集样本: {len(train_x)} | 测试集样本: {len(test_x)} | multi_step={args.multi_step}")

    train_loader, test_loader = build_dataloaders(
        train_x,
        train_y,
        test_x,
        batch_size=256,
        pad_token=DEFAULT_CODE_PAD_TOKEN,
    )

    model = RQKMeansTransformer()
    model = train_model(model, train_loader, epochs=5, lr=0.001)

    code_to_cities = build_code_to_cities(city_to_codes, train_set)
    fallback = top_city_ids_from_train(train_set, k=4)
    all_predictions = predict_top4_cities(
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
    submission_path = out_dir / f"submission_rqkmeans_transformer_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"✅ {submission_path} 已生成！")
    print_accuracy_at_4_report(
        submission_df,
        skip=args.skip_eval,
        ground_truth_path=args.ground_truth,
    )


if __name__ == "__main__":
    main()

from datetime import datetime
from pathlib import Path
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.dataset import build_dataloaders
from src.preprocessing import (
    build_code_to_cities,
    build_final_dataset,
    build_rq_codebook,
    create_mutliple_sequences,
    create_trip_sequences,
    create_multiple_sequences,
    train_word2vec,
)
from src.rqvae_gru import RQVAEPredictor
from src.rqvae_train_infer import predict_top4_cities, train_model
from src.rqvae_transformer import RQVAETransformer

def main():
    # root_dir = Path(__file__).resolve().parents[2]
    # data_dir = root_dir / "data"
    output_dir = "/Users/althealam/Desktop/GitHub/Generative-Retrieval-for-Multi-Destination-Trips-via-RQ-VAE/output"
    # output_dir.mkdir(parents=True, exist_ok=True)

    train_set = pd.read_csv("/Users/althealam/Desktop/GitHub/Generative-Retrieval-for-Multi-Destination-Trips-via-RQ-VAE/data/train_set.csv")
    test_set = pd.read_csv("/Users/althealam/Desktop/GitHub/Generative-Retrieval-for-Multi-Destination-Trips-via-RQ-VAE/data/test_set.csv")

    print("正在聚合行程序列...")
    train_trips = create_mutliple_sequences(train_set)
    test_trips = create_mutliple_sequences(test_set)

    print("正在训练 Word2Vec...")
    w2v = train_word2vec(train_trips, vector_size=64, window=5)

    print("正在构建 RQ-VAE 语义索引 (Residual Quantization)...")
    city_to_codes = build_rq_codebook(train_set, w2v, n_clusters=32, random_state=42)

    train_x, train_y = build_final_dataset(train_trips, city_to_codes, is_test=False)
    test_x, _ = build_final_dataset(test_trips, city_to_codes, is_test=True)
    print("✅ 数据集构建完成！")
    print(f"训练集样本: {len(train_x)} | 测试集样本: {len(test_x)}")

    train_loader, test_loader = build_dataloaders(train_x, train_y, test_x, batch_size=256)
    
    model = RQVAETransformer()
    model = train_model(model, train_loader, epochs=5, lr=0.001)

    code_to_cities = build_code_to_cities(city_to_codes, train_set)
    all_predictions = predict_top4_cities(model, test_loader, code_to_cities)

    submission_df = pd.DataFrame(
        all_predictions,
        columns=["city_id_1", "city_id_2", "city_id_3", "city_id_4"],
    )
    submission_df.insert(0, "utrip_id", test_trips["utrip_id"].tolist())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = os.path.join(output_dir, f"submission_{timestamp}.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"✅ {submission_path} 已生成！")


if __name__ == "__main__":
    main()

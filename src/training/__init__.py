from src.training.code_predict import predict_top4_with_codebook, train_code_transformer
from src.training.embedding import recommend_top4_cities, train_city_transformer

# Backward-compatible names for `from src.training import ...`
predict_top4_cities_rqkmeans = predict_top4_with_codebook
train_rqkmeans_model = train_code_transformer
predict_top4_cities_from_rqvae = predict_top4_with_codebook
train_rqvae_model = train_code_transformer

__all__ = [
    "train_city_transformer",
    "recommend_top4_cities",
    "train_code_transformer",
    "predict_top4_with_codebook",
    "train_rqvae_model",
    "predict_top4_cities_from_rqvae",
    "train_rqkmeans_model",
    "predict_top4_cities_rqkmeans",
]

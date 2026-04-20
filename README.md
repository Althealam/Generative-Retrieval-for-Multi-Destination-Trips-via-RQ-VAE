# 🌍 Generative Retrieval for Multi-Destination Trips via RQ-VAE

## 🧭 Overview

Multi-destination trips involve travelers visiting multiple cities in a single journey. This project aims to predict the next destination city for incomplete multi-destination trips using advanced machine learning techniques.

## 🗂️ Dataset

The project uses the **Booking.com Multi-Destination Trips Dataset** from the WSDM WebTour 2021 Challenge.

The training dataset consists of over a million of anonymized hotel reservations, based on real data, with the following features: 
* user_id: User ID
* check-in: reservation check-in date
* checkout: reservation check-out date
* affiliate_id: An anonymized ID of affiliate channels where the booker came from (e.g., direct, some third party referrals, paid search engine, etc.)
* device_class: desktop/mobile
* booker_country: Country from which the reservation was made (anonymized)
* hotel_country: Country of the hotel (anonymized)
* city_id: city id of the hotel's city (anonymized)
* utrip_id: Unique identification


### 📊 Dataset Statistics
- **Training Set**: 1,166,835 bookings
- **Test Set**: 378,667 bookings (with 70,662 trips to predict)
- **Features**: user_id, checkin, checkout, city_id, device_class, affiliate_id, booker_country, hotel_country, utrip_id
- **Target**: Predict the final destination city for incomplete trips

## 📚 References
### 💻 Compute Resource
* GPU: NVIDIA GeForce RTX 5090, 1
* CPU: 25 Core

### Dataset
- **Data Source**: [Booking.com Multi-Destination Trips Dataset](https://github.com/bookingcom/ml-dataset-mdt)
- **Paper**: [Multi-Destination Trip Dataset](https://dl.acm.org/doi/10.1145/3404835.3463240)
- **Challenge**: Booking.com WSDM WebTour 2021 Challenge
- **Conference**: [WSDM 2021](https://ceur-ws.org/Vol-2855/)


## 🙏 Acknowledgments

- Booking.com for providing the multi-destination trips dataset
- WSDM WebTour 2021 Challenge organizers

## 🧪 Experiments
Experiment notes: https://my.feishu.cn/wiki/ICjgw24P8iIb9rkrIVJc17AEnBc?fromScene=spaceOverview
### 2026/4/7
* Use word2vec to generate the sparse city_id embeddings
* Use rq-vae to generate the discrete city_id representations by using word2vec embeddings
* Use GRU to predict the next city_id 
* Score: 0.33884
### 2026/4/8
* Use transformer+rq-kmeans+word2vec to predict the next city_id, and its score is 0.33429
* Increase the embedding_dim from 128 to 256, but the score is decreasing
* Drop the RQ-KMeans, use embedding table, and its score is 0.44354815884067816
### 2026/4/9
* Find the reason why RQ-KMeans performance isn't great, and improve the word2vec training, but it doesn't have a better performance than normal embedding
* Add RQ-VAE to encode the city_id, and test the accuracy@4, but it only have 0.249
### 2026/4/11
* Update feature engineering
* Add a multi-step training, which means that we can train A-B-C with A-B and A-B-C, so it will have more training data
* Embedding model performance is 0.45, and rqvae with transformer performance is 0.3
### 2026/4/12
* Use autodl to train the model with GPU, and turn on the multi step options. After adding multi-step, the embedding model performance improve to 0.4559.
* Add multi-step options to RQVAE+Transformer and RQKmeans+Transformer, with RQVAE 0.325861 and RQKmeans 0.282471, which can prove that multi-step really improve the performance.
### 2026/4/13
* Reconstruct the code, and add use_context for rqkmeans and rqvae
* Feature Engineering like example github
    - Embedding: 0.458422
    - RQKMeans: 0.272268
    - RQVAE: 0.327814
* Add GRU for rqkmeans and rqvae
    - RQKMeans: 0.329144
    - RQVAE: 0.343622
### 2026/4/14
* Fix the hidden state problem for both gru and transformer, and it turns out that it improve the performance especially in transformer architecture
* Add GRU for embedding model
* Transformer
    - Embedding: 0.482098
    - RQKMeans: 0.306643
    - RQVAE: 0.333744
* GRU
    - RQKMeans: 0.309035
    - RQVAE: 0.348745
    - Embedding: 0.489117
* Add choice for hidden state which will be feed into the classification network: CLS/Last Hidden/Mean (this is only for transformer, and now I just implement on embedding model)
    - last: 0.485508
    - CLS: 0.076774
    - mean: 0.485508
### 2026/4/15
* Add geograpy features into the embedding, rqvae, rqkmeans model. The features are: last_hotel_country, unique_hotel_countries, cross_border_count, cross_border_ratio. And the performance are like:
    - Embedding with transformer: 0.487093
    - RQVAE with transformer: 0.338428
    - RQKMeans with transformer: 0.290920
### 2026/4/16
* Drop causal mask for embedding model
    - Embedding with transformer: 0.485296 
* Add affiliate_id
    - Embedding with transformer: 0.483202
    - RQVAE: 0.341598
    - RQKMeans: 0.304379
* Change the type of device_class: before that we use "first", which means that we just get the first trip device type, but now we use the list of device_class
### 2026/4/20
* Add semantic ID as side info into the embedding model
    - Embedding with rqvae: 0.484334
    - Embedding with rqkmeans: 0.484093
* Feature confusion: add gate mechanism to combine context features and sequence last hidden states feature
    - Embedding: 0.480598

### ✨ Useful Tricks (from experiments)
- **Multi-step training is consistently helpful**: turning on `--multi_step` improves all three pipelines (Embedding / RQVAE / RQKMeans), especially when training data is sparse at longer sequence lengths.
- **Hidden-state extraction matters**: fixing last-hidden extraction logic gave clear gains in both Transformer and GRU variants.
- **Transformer pooling choice is critical**: `last` and `mean` work much better than `cls` in the current embedding setup.
- **Geography features are useful**: `last_hotel_country`, `unique_hotel_countries`, `cross_border_count`, `cross_border_ratio` improved performance.
- **Direct city embedding baseline is strong**: embedding-based next-city classification outperforms current RQ code-based routes.
- **Semantic side info is not always additive**: adding RQ semantic IDs can help slightly, but naive gate fusion may hurt without extra tuning.

## 🏗️ Architecture and Performance Comparison

| Pipeline | Input Representation | Prediction Target | Decoder Strategy | Transformer Best Accuracy@4 | GRU Best Accuracy@4 |
|---|---|---|---|---:|---:|
| Embedding | City prefix token sequence + context features | Direct next `city_id` token | Single classifier over city vocabulary | `0.487093` | `0.489117` |
| RQVAE | RQ code sequence + context features | Next `(code1, code2)` pair | Map top code pairs to cities via `code_to_cities` | `0.338428` | `0.348745` |
| RQKMeans | Residual k-means code sequence + context features | Next `(code1, code2)` pair | Map top code pairs to cities via `code_to_cities` | `0.306643` | `0.329144` |

### 📝 Pipeline Notes
- **Embedding**: predicts city directly; currently the strongest route in this project.
- **RQVAE / RQKMeans**: predict semantic code pairs first, then decode back to candidate cities.

## 🚀 How To Run

Use the unified entry script:

```bash
./scripts/run_train.sh <embedding|rqvae|rqkmeans> [extra args...]
```

### 1) 🔹 Train embedding model

```bash
./scripts/run_train.sh embedding --multi_step
```

Example with semantic side info and gate fusion:

```bash
./scripts/run_train.sh embedding \
  --multi_step \
  --fusion gate \
  --semantic_source rqkmeans
```

Or use RQVAE semantic mapping:

```bash
./scripts/run_train.sh embedding \
  --multi_step \
  --semantic_source rqvae \
  --semantic_mapping_path "/root/gr/Generative-Retrieval-for-Multi-Destination-Trips/output/rqvae/city_to_codes_rqvae_20260409_110222.json"
```

### 2) 🔹 Train RQVAE code prediction model

```bash
./scripts/run_train.sh rqvae --multi_step
```

Use a specific mapping file:

```bash
./scripts/run_train.sh rqvae --multi_step --mapping_path "/path/to/city_to_codes_rqvae_xxx.json"
```

### 3) 🔹 Train RQKMeans code prediction model

```bash
./scripts/run_train.sh rqkmeans --multi_step
```

### 🔧 Optional: run pipeline-specific scripts directly

- `./scripts/run_train_model_with_embedding.sh ...`
- `./scripts/run_train_model_with_rqvae.sh ...`
- `./scripts/run_train_model_with_rqkmeans.sh ...`
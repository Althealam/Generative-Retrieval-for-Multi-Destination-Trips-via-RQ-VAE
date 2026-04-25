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

### 📝 Dataset
- **Data Source**: [Booking.com Multi-Destination Trips Dataset](https://github.com/bookingcom/ml-dataset-mdt)
- **Paper**: [Multi-Destination Trip Dataset](https://dl.acm.org/doi/10.1145/3404835.3463240)
- **Challenge**: Booking.com WSDM WebTour 2021 Challenge
- **Conference**: [WSDM 2021](https://ceur-ws.org/Vol-2855/)

## 🧪 Experiments
Experiment notes: [Feishu Wiki](https://my.feishu.cn/wiki/ICjgw24P8iIb9rkrIVJc17AEnBc?fromScene=spaceOverview)

### 🗓️ Timeline

#### 2026-04-07
- Trained Word2Vec city embeddings and RQ-VAE discrete city representations.
- Used GRU for next-city prediction.
- Score: `0.33884`.

#### 2026-04-08
- Tested Transformer + RQ-KMeans + Word2Vec.
- Result: `0.33429`.
- Increased embedding dimension from `128` to `256` (performance decreased).
- Dropped RQ-KMeans and switched to direct embedding table.
- Result after switch: `0.443548`.

#### 2026-04-09
- Investigated weak RQ-KMeans performance and improved Word2Vec training.
- RQ-KMeans still underperformed compared with direct embedding baseline.
- Added RQ-VAE encoding for city IDs.
- RQ-VAE Accuracy@4: `0.249`.

#### 2026-04-11
- Updated feature engineering.
- Introduced multi-step training (prefix-to-next-city expansion).
- Results:
  - Embedding: `0.45`
  - RQVAE + Transformer: `0.3`

#### 2026-04-12
- Enabled GPU training on AutoDL and turned on multi-step.
- Embedding improved to `0.4559`.
- Added multi-step to RQVAE+Transformer and RQKMeans+Transformer:
  - RQVAE: `0.325861`
  - RQKMeans: `0.282471`

#### 2026-04-13
- Refactored code and added context usage for RQKMeans / RQVAE.
- Feature engineering update:
  - Embedding: `0.458422`
  - RQKMeans: `0.272268`
  - RQVAE: `0.327814`
- Added GRU for RQKMeans / RQVAE:
  - RQKMeans: `0.329144`
  - RQVAE: `0.343622`

#### 2026-04-14
- Fixed hidden-state extraction for GRU and Transformer (notable gains, especially Transformer).
- Added GRU for embedding model.
- Results by architecture:
  - Transformer:
    - Embedding: `0.482098`
    - RQKMeans: `0.306643`
    - RQVAE: `0.333744`
  - GRU:
    - RQKMeans: `0.309035`
    - RQVAE: `0.348745`
    - Embedding: `0.489117`
- Added Transformer pooling options for embedding model:
  - `last`: `0.485508`
  - `cls`: `0.076774`
  - `mean`: `0.485508`

#### 2026-04-15
- Added geography features (`last_hotel_country`, `unique_hotel_countries`, `cross_border_count`, `cross_border_ratio`).
- Results:
  - Embedding + Transformer: `0.487093`
  - RQVAE + Transformer: `0.338428`
  - RQKMeans + Transformer: `0.290920`

#### 2026-04-16
- Dropped causal mask in embedding Transformer:
  - Embedding + Transformer: `0.485296`
- Added `affiliate_id`:
  - Embedding + Transformer: `0.483202`
  - RQVAE: `0.341598`
  - RQKMeans: `0.304379`
- Updated `device_class` handling from a single first value to full sequence list.

#### 2026-04-20
- Added semantic IDs as side information in embedding model:
  - Embedding + RQVAE semantic IDs: `0.484334`
  - Embedding + RQKMeans semantic IDs: `0.484093`
- Added gate fusion for context + sequence hidden state:
  - Embedding: `0.480598`

### ✨ Useful Tricks (from experiments)
- **Multi-step training is consistently helpful**: turning on `--multi_step` improves all three pipelines (Embedding / RQVAE / RQKMeans), especially when training data is sparse at longer sequence lengths.
- **Hidden-state extraction matters**: fixing last-hidden extraction logic gave clear gains in both Transformer and GRU variants.
- **Transformer pooling choice is critical**: `last` and `mean` work much better than `cls` in the current embedding setup.
- **Geography features are useful**: `last_hotel_country`, `unique_hotel_countries`, `cross_border_count`, `cross_border_ratio` improved performance.
- **Direct city embedding baseline is strong**: embedding-based next-city classification outperforms current RQ code-based routes.
- **Semantic side info is not always additive**: adding RQ semantic IDs can help slightly, but naive gate fusion may hurt without extra tuning.

## 🏗️ Architecture and Performance Comparison

### Embedding Model Architecture

![Embedding Architecture](assets/embedding-architecture.png)

### RQVAE Pipeline Architecture 

![RQVAE Architecture](assets/rqvae-architecture.png)

### RQKMeans Pipeline Architecture

![RQKMeans Architecture](assets/rqkmeans-architecture.png)

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

## 🧪 Testing

> **Note**: The test suite in the `tests/` directory was generated and debugged by [Claude Code](https://claude.ai/code), Anthropic's AI coding assistant, ensuring comprehensive coverage of all components.

Run unit tests to verify code correctness:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

**Test Coverage**:
- ✅ Feature engineering functions
- ✅ All model architectures (forward pass, edge cases)
- ✅ Data loading and batching
- ✅ Evaluation metrics
- ✅ Utility functions
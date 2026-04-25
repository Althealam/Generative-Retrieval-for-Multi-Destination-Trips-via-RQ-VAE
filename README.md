# 🌍 Generative Retrieval for Multi-Destination Trips

Predicting the next city for incomplete multi-destination trips using sequence models and semantic code-based retrieval pipelines.

## TL;DR (For Instructor/Reviewer)
- **Task**: Given a trip prefix, predict the next destination city (evaluated by Accuracy@4).
- **Best result in this repo**: **Embedding + GRU = `0.489117`** (Embedding + Transformer = `0.487093`).
- **Main conclusion**: Direct city-token modeling is currently stronger than RQ code-pair pipelines (RQ-VAE / RQKMeans) on this dataset.
- **Key lesson**: Feature engineering and hidden-state extraction have larger impact than architecture complexity alone.

## 1) Problem Setup
Multi-destination travel prediction is framed as next-step sequence prediction:
- Input: city prefix + contextual features
- Output: top-4 candidate next cities
- Metric: Accuracy@4

Dataset source: Booking.com Multi-Destination Trips (WSDM WebTour 2021).

## 2) Dataset
### Statistics
- **Train**: 1,166,835 booking rows
- **Test**: 378,667 booking rows (70,662 trips to complete)

### Core columns
`user_id`, `checkin`, `checkout`, `affiliate_id`, `device_class`, `booker_country`, `hotel_country`, `city_id`, `utrip_id`.

### Trip-level preprocessing
Rows are sorted by check-in time and grouped by `utrip_id` to form sequences with temporal/context features.

## 3) Method Overview
This repository implements three pipelines:

1. **Embedding pipeline (best-performing)**  
   - Directly predicts next `city_id` token  
   - Backbones: Transformer / GRU  
   - Uses context features + optional semantic side IDs

2. **RQVAE pipeline**  
   - Converts city IDs to two-level semantic codes via residual vector quantization  
   - Predicts next `(code1, code2)` then decodes back to city candidates

3. **RQKMeans pipeline**  
   - Uses residual k-means codebook over Word2Vec city embeddings  
   - Same next code-pair prediction and decoding pattern

## 4) Main Results
| Pipeline | Input Representation | Prediction Target | Transformer Best A@4 | GRU Best A@4 |
|---|---|---|---:|---:|
| Embedding | City prefix + context | Next `city_id` | `0.487093` | `0.489117` |
| RQVAE | RQ code sequence + context | Next `(code1, code2)` | `0.338428` | `0.348745` |
| RQKMeans | Residual k-means code sequence + context | Next `(code1, code2)` | `0.306643` | `0.329144` |

### Interpretation
- Direct token prediction is currently the strongest approach.
- Code-based routes are meaningful for generative retrieval framing, but currently underperform in end accuracy.

## 5) What Helped Most
- Multi-step training (`--multi_step`) generally improved results.
- Correct hidden-state extraction gave clear gains for both GRU and Transformer.
- Transformer pooling: `last` / `mean` worked well; `cls` failed in this setting.
- Geography-aware features (`last_hotel_country`, cross-border stats) improved embedding performance.

## 6) Quick Start
Use unified launcher:

```bash
./scripts/run_train.sh <embedding|rqvae|rqkmeans> [extra args...]
```

Examples:

```bash
# Embedding (recommended baseline)
./scripts/run_train.sh embedding --multi_step

# RQVAE
./scripts/run_train.sh rqvae --multi_step

# RQKMeans
./scripts/run_train.sh rqkmeans --multi_step
```

Optional direct scripts:
- `./scripts/run_train_model_with_embedding.sh ...`
- `./scripts/run_train_model_with_rqvae.sh ...`
- `./scripts/run_train_model_with_rqkmeans.sh ...`

## 7) Repository Structure (High Level)
- `src/features`: trip aggregation, context features, codebook-related feature transforms
- `src/models`: embedding / rqvae / rqkmeans model implementations
- `src/training`: training loops and inference logic
- `src/datasets`: dataloaders/collators
- `scripts`: runnable training entry points

## 8) Testing
```bash
pip install pytest pytest-cov
pytest
pytest --cov=src --cov-report=html
```

Coverage includes feature engineering, model forward passes, dataloaders, evaluation, and utilities.

---

## Appendix A: Architecture Figures
### Embedding
![Embedding Architecture](assets/embedding-architecture.png)

### RQVAE
![RQVAE Architecture](assets/rqvae-architecture.png)

### RQKMeans
![RQKMeans Architecture](assets/rqkmeans-architecture.png)

## Appendix B: Experiment Timeline (Detailed Log)
### 2026-04-07
- Word2Vec + RQ-VAE codes + GRU baseline: `0.33884`.

### 2026-04-08
- Transformer + RQKMeans: `0.33429`.
- Switched to direct embedding table: `0.443548`.

### 2026-04-09
- Improved Word2Vec for RQKMeans, still weaker than embedding.
- Added RQ-VAE encoding: `0.249`.

### 2026-04-11
- Added multi-step and feature updates.
- Embedding: `0.45`, RQVAE+Transformer: `0.3`.

### 2026-04-12
- AutoDL GPU + multi-step:
  - Embedding: `0.4559`
  - RQVAE: `0.325861`
  - RQKMeans: `0.282471`

### 2026-04-13
- Refactor + context extension:
  - Embedding: `0.458422`
  - RQKMeans: `0.272268`
  - RQVAE: `0.327814`
- Added GRU:
  - RQKMeans: `0.329144`
  - RQVAE: `0.343622`

### 2026-04-14
- Fixed hidden extraction:
  - Transformer: Embedding `0.482098`, RQKMeans `0.306643`, RQVAE `0.333744`
  - GRU: RQKMeans `0.309035`, RQVAE `0.348745`, Embedding `0.489117`
- Pooling test:
  - `last`: `0.485508`, `mean`: `0.485508`, `cls`: `0.076774`

### 2026-04-15
- Added geography features:
  - Embedding+Transformer: `0.487093`
  - RQVAE+Transformer: `0.338428`
  - RQKMeans+Transformer: `0.290920`

### 2026-04-16
- Dropped causal mask in embedding Transformer: `0.485296`
- Added `affiliate_id`:
  - Embedding: `0.483202`
  - RQVAE: `0.341598`
  - RQKMeans: `0.304379`

### 2026-04-20
- Added semantic side IDs in embedding:
  - +RQVAE IDs: `0.484334`
  - +RQKMeans IDs: `0.484093`
- Gate fusion trial: `0.480598`

## References
### Compute Resource
- GPU: NVIDIA GeForce RTX 5090 (1)
- CPU: 25 cores

### Dataset and papers
- Dataset: [Booking.com Multi-Destination Trips Dataset](https://github.com/bookingcom/ml-dataset-mdt)
- Paper: [Multi-Destination Trip Dataset](https://dl.acm.org/doi/10.1145/3404835.3463240)
- Challenge: Booking.com WSDM WebTour 2021
- Conference: [WSDM 2021](https://ceur-ws.org/Vol-2855/)

### Generative Retrieval / Quantization / Recommendation (Related Work)
- **VQ-VAE**: van den Oord et al., *Neural Discrete Representation Learning* (NeurIPS 2017)  
  [https://arxiv.org/abs/1711.00937](https://arxiv.org/abs/1711.00937)
- **Residual Quantization for VQ**: Zeghidour et al., *SoundStream: An End-to-End Neural Audio Codec* (TASLP 2021)  
  [https://arxiv.org/abs/2107.03312](https://arxiv.org/abs/2107.03312)
- **Product/Residual Quantization for ANN Retrieval**: Jegou et al., *Product Quantization for Nearest Neighbor Search* (TPAMI 2011)  
  [https://hal.inria.fr/inria-00514462](https://hal.inria.fr/inria-00514462)
- **Residual Vector Quantization in retrieval systems**: Babenko and Lempitsky, *The Inverted Multi-Index* (CVPR 2012)  
  [https://ieeexplore.ieee.org/document/6247798](https://ieeexplore.ieee.org/document/6247798)
- **Transformer-based Sequential Recommendation**: Sun et al., *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer* (CIKM 2019)  
  [https://arxiv.org/abs/1904.06690](https://arxiv.org/abs/1904.06690)
- **Generative Recommendation**: Rajput et al., *Recommender Systems with Generative Retrieval* (NeurIPS 2023)  
  [https://arxiv.org/abs/2305.05065](https://arxiv.org/abs/2305.05065)

### Notes on how these connect to this project
- RQVAE pipeline is inspired by discrete latent code learning (VQ-VAE family) and residual quantization.
- RQKMeans pipeline is closer to classical quantization-based indexing (PQ/RQ style).
- Embedding+Transformer/GRU route aligns with modern sequential recommendation literature.
- The project evaluates whether semantic code prediction + decoding can compete with direct token prediction.
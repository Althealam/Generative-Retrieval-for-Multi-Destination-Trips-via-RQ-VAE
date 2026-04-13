# Generative Retrieval for Multi-Destination Trips via RQ-VAE

This project implements a generative retrieval approach for predicting multi-destination trips using Residual Quantized Variational Autoencoder (RQ-VAE), based on the Booking.com Multi-Destination Trips dataset.

## Overview

Multi-destination trips involve travelers visiting multiple cities in a single journey. This project aims to predict the next destination city for incomplete multi-destination trips using advanced machine learning techniques.

## Dataset

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


### Dataset Statistics
- **Training Set**: 1,166,835 bookings
- **Test Set**: 378,667 bookings (with 70,662 trips to predict)
- **Features**: user_id, checkin, checkout, city_id, device_class, affiliate_id, booker_country, hotel_country, utrip_id
- **Target**: Predict the final destination city for incomplete trips

## References

### Dataset
- **Data Source**: [Booking.com Multi-Destination Trips Dataset](https://github.com/bookingcom/ml-dataset-mdt)
- **Paper**: [Multi-Destination Trip Dataset](https://dl.acm.org/doi/10.1145/3404835.3463240)
- **Challenge**: Booking.com WSDM WebTour 2021 Challenge
- **Conference**: [WSDM 2021](https://ceur-ws.org/Vol-2855/)


## Acknowledgments

- Booking.com for providing the multi-destination trips dataset
- WSDM WebTour 2021 Challenge organizers

## Experiments
record note: https://my.feishu.cn/wiki/ICjgw24P8iIb9rkrIVJc17AEnBc?fromScene=spaceOverview
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
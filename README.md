CVAE-GM: Contrastive Variational Autoencoder with Gaussian Mixture Prior for SSD Subtyping

This repository contains the PyTorch implementation of the CVAE-GM model, designed for self-supervised subtyping of Schizophrenia Spectrum Disorders (SSD) using EEG and clinical data. The model integrates:

Multi-frequency contrastive representation learning

Gaussian mixture prior in latent space

Self-supervised clustering for discovering SSD subtypes

🧠 Key Features

Input Modalities:

Absolute EEG power across six frequency bands: delta, slow theta, fast theta, alpha, beta, gamma

Clinical scales: Neuro-11, HAMD, HAMA, PSQI

Model Architecture:

Encoder/Decoder: 3 fully-connected layers

Latent space dimension: 64

Gaussian mixture prior with 5 components

Frequency-aware contrastive module with two-layer MLP (temperature = 0.07)

Loss Functions:

Reconstruction loss

KL divergence loss

GMM prior loss

Frequency-based contrastive loss

Weighted sum: Loss = recon + 0.5*(KL + GMM_KL) + 1.0*contrastive

Training Strategy:

Optimizer: Adam (lr=1e-3)

Batch size: 128

Epochs: 200

ReduceLROnPlateau (patience=10, factor=0.5)

Early stopping (patience=20)

5-fold cross-validation

Evaluation Metrics:

ARI, NMI, V-measure, silhouette score, accuracy, sensitivity, specificity

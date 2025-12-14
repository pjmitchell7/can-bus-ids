# CAN Bus Intrusion Detection using Deep Learning

This repository contains the code for my CAN bus intrusion detection project, which explores the use of an unsupervised deep learning autoencoder to detect malicious activity on in-vehicle networks.

The goal of this project was not to build a fully production-ready IDS, but to design a correct, interpretable, and real-time-capable baseline that learns normal CAN traffic behavior and flags deviations using reconstruction error. The work emphasizes preprocessing correctness, evaluation realism, and latency feasibility alongside detection performance.

This repository supports the experiments and results presented in my paper:
**“Deep Learning–Based Intrusion Detection for CAN Bus Security in Autonomous Vehicles.”**

## Project Overview

Modern vehicles rely on the Controller Area Network (CAN) bus to coordinate safety-critical systems, yet the protocol lacks authentication and encryption. Rather than relying on fixed rules or known attack signatures, this project uses an autoencoder trained only on benign CAN traffic. During inference, windows that reconstruct poorly are treated as anomalous.

Key design choices:
- Unsupervised training on benign data only
- Window-based representation of CAN traffic
- Reconstruction error as an anomaly score
- Percentile-based thresholding for deployment realism
- Explicit evaluation of latency and throughput

## Dataset

Experiments use the publicly available **Car Hacking Dataset (OCSLab)**, which includes:
- Benign CAN traffic
- Denial of Service (DoS) attacks
- Fuzzy injection attacks
- Gear spoofing attacks
- RPM spoofing attacks

The dataset is not included in this repository due to size. Instructions for obtaining it are provided below.

Dataset source:
https://ocslab.hksecurity.net/Datasets/car-hacking-dataset

## Repository Structure
├── src/
│ ├── preprocess.py # CAN parsing, normalization, windowing
│ ├── train.py # Autoencoder training (benign only)
│ ├── evaluate.py # Thresholding and evaluation logic
│ ├── model.py # Autoencoder architecture
│ └── utils.py # Helper functions
├── experiments/
│ └── run_*/ # Saved configs and evaluation outputs
├── requirements.txt
├── README.md


Large artifacts such as raw datasets, trained models, and NumPy window dumps are intentionally excluded and regenerated locally.

## Preprocessing Summary

Each CAN message is converted into a fixed numerical representation consisting of:
- 8 payload bytes (decoded from hex)
- Data Length Code (DLC)
- Encoded CAN ID

Messages are grouped into overlapping windows of 64 frames with a hop size of 32. Each window is normalized using statistics computed from benign training data only.

Window-level labels for evaluation are derived using an “any-attack-in-window” rule, meaning a window is considered malicious if it contains at least one injected frame. This matches realistic IDS behavior and avoids optimistic evaluation.

## Model

The model is a fully connected autoencoder trained to minimize mean squared reconstruction error on benign windows.

- Input dimension: 640 (64 × 10)
- Symmetric encoder/decoder
- Narrow latent bottleneck to prevent memorization
- MSE reconstruction loss
- Adam optimizer

The model is intentionally simple to keep inference latency low and behavior interpretable.

## Detection and Thresholding

During inference, each window produces a reconstruction error that acts as a continuous anomaly score.

Two thresholding strategies are implemented:
- Percentile-based thresholds derived from benign training data (deployment-oriented)
- Validation F1-optimized thresholds (diagnostic only)

Most reported results use a conservative 99th percentile threshold to prioritize precision and reduce false positives.

## Results Summary

The system achieves:
- Very high precision across validation and test sets
- Moderate recall, especially for subtle spoofing attacks
- Sub-millisecond inference latency on GPU
- Throughput well above real CAN bus requirements

These results demonstrate that reconstruction-based IDS models are viable for real-time in-vehicle deployment, while also highlighting the limitations of static window approaches.

## Running the Code

### 1. Install dependencies
```bash
pip install -r requirements.txt

2. Preprocess data
python src/preprocess.py --input_dir PATH_TO_CAN_CSVS --output_dir data/

3. Train model
python src/train.py

4. Evaluate
python -m src.evaluate



Evaluation outputs include precision, recall, F1 score, flagged window rates, and latency measurements.

## Notes on Reproducibility

Care was taken to avoid common pitfalls such as:

Label/window misalignment

Normalization leakage across splits

Optimistic threshold selection

All preprocessing decisions were validated against realistic deployment constraints.

## Future Work

This repository represents a baseline system. Natural next steps include:

Sequence-aware models (LSTM, Transformer)

Attention-based feature weighting

Multi-vehicle training for better generalization

Hybrid anomaly + rule-based detection

## Author

Paul J. Mitchell
Department of Computer Science
College of William & Mary
pjmitchell@wm.edu

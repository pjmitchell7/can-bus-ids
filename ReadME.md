# CAN Bus Intrusion Detection using Deep Learning

This repository contains my final project for **CSCI 680 – Autonomous Driving & Connected Mobility** at the College of William & Mary.

The goal of this project is to explore whether deep learning models can detect malicious activity on a vehicle’s **Controller Area Network (CAN) bus** under realistic constraints. Instead of relying on fixed rules or known attack signatures, the system learns what *normal* CAN traffic looks like and flags deviations as anomalies.

The current implementation focuses on an **unsupervised autoencoder baseline**, with an emphasis on correctness, interpretability, and real-time feasibility rather than chasing benchmark numbers.

---

## Project Overview

Modern vehicles rely on dozens of Electronic Control Units (ECUs) that communicate over the CAN bus. While CAN is efficient and reliable, it lacks authentication and encryption, meaning any compromised node can inject messages that appear legitimate.

This project builds an intrusion detection system (IDS) that:
- Learns normal CAN communication patterns from benign data
- Detects anomalous behavior using reconstruction error
- Operates fast enough to meet real-time constraints
- Avoids reliance on labeled attack data during training

The system is evaluated using the **Car Hacking Dataset**, which includes both obvious attacks (e.g., flooding) and subtle spoofing attacks (e.g., RPM and Gear manipulation).

---

## Repository Structure
CAN_Bus_Security/
│
├── src/
│ ├── preprocess.py # CAN log parsing, normalization, windowing
│ ├── evaluate.py # Reconstruction error, thresholding, metrics
│ └── model.py # Autoencoder definition and training logic
│
├── tools/
│ └── add_window_labels.py # Aligns per-message labels to window labels
│
├── experiments/
│ └── run_*/ # Saved experiment configs and results
│
├── data/ # (Ignored) Raw and processed CAN data
│
└── README.md

The `data/` directory is intentionally not included in the repository due to dataset size and licensing. See below for dataset details.

---

## Dataset

This project uses the **Car Hacking Dataset** from OCSLab:

https://ocslab.hksecurity.net/Datasets/car-hacking-dataset

The dataset contains:
- One benign CAN traffic log
- Four attack types:
  - Denial of Service (DoS)
  - Fuzzy injection
  - Gear spoofing
  - RPM spoofing

Training is performed using **benign traffic only**, consistent with an unsupervised anomaly detection setup. Attack data is used strictly for validation and testing.

---

## Preprocessing Pipeline

Raw CAN logs are converted into model-ready inputs through the following steps:

1. **Feature extraction**
   - 8 payload bytes
   - DLC (Data Length Code)
   - Encoded CAN ID

2. **Normalization**
   - Z-score normalization using statistics from benign training data only
   - Prevents byte values from dominating reconstruction loss

3. **Windowing**
   - 64 CAN frames per window
   - Stride of 32 (50% overlap)
   - Each window represents ~100 ms of CAN activity

4. **Label alignment**
   - A window is labeled anomalous if *any* message inside the window is malicious
   - This reflects realistic safety requirements

Earlier versions of the pipeline suffered from window-label misalignment, which produced unrealistically high metrics. These issues were corrected, and all reported results reflect properly aligned evaluation.

---

## Model

The baseline model is a **fully connected autoencoder** trained only on benign CAN windows.

Key properties:
- Input dimension: 640 (64 frames × 10 features)
- Bottleneck forces compression of normal behavior
- Mean squared error used as reconstruction loss
- No attack data seen during training

The model does not attempt to classify attack types. Instead, it flags windows that do not conform to learned normal patterns.

---

## Detection and Thresholding

Reconstruction error serves as the anomaly score.

Two thresholding strategies are explored:
- **Training percentile threshold** (deployment-oriented, high precision)
- **Validation F1-optimized threshold** (diagnostic only)

Percentile-based thresholds are used to avoid reliance on absolute error values and to improve robustness across environments.

---

## Results Summary

Key findings:
- Precision is consistently near perfect (>99%)
- Recall is moderate, especially for subtle spoofing attacks
- Inference latency is sub-millisecond on GPU
- Threshold choice strongly affects alert rate

The model reliably detects obvious attacks but struggles with stealthy attacks that closely mimic normal behavior. This is an expected limitation of static-window reconstruction models and motivates future sequence-aware approaches.

---

## Limitations and Future Work

This project intentionally focuses on a simple, interpretable baseline. Identified limitations include:
- Lack of explicit temporal modeling
- Payload byte dominance in reconstruction loss
- Sensitivity to threshold selection
- Limited dataset diversity

Future directions include:
- LSTM or Transformer-based sequence models
- Feature reweighting to emphasize message identity and timing
- Training on CAN logs from multiple vehicles and environments
- Hybrid systems combining anomaly detection with lightweight rules

---

## Reproducibility Notes

- All preprocessing statistics are computed from training data only
- CAN ID mappings are saved and reused across splits
- Evaluation uses window-level labels derived from message-level annotations
- Results reported in the paper and slides correspond to committed code

---

## Author

**Paul J. Mitchell**  
M.S. Computer Science  
College of William & Mary  
pjmitchell@wm.edu

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


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


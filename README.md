# GRAVISCOPE AI – Gravitational Wave Anomaly Hunter

This project detects anomalies in gravitational wave strain data using deep spatio-temporal learning. Data is sourced from LIGO's O3a 4kHz public release (H1 detector).

## 📁 Folder Structure

D:
└── GRAVISCOPE_AI
├── data
│ ├── raw\ # Original HDF5 files from LIGO
│ └── processed\ # Normalized .npy files for model input
├── src
│ ├── preprocess.py # Extract & normalize strain data
│ └── eda.py # Visualize waveform & spectrogram
├── outputs\ # Saved figures
└── models\ # Deep learning models (CNN+Bi-GRU+Attn)

## ▶️ How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Preprocess `.hdf5` files:
python src/preprocess.py

3. Run visualization:
python src/eda.py

## 📌 Dataset

- Source: [LIGO Open Science Center – O3a 4kHz](https://www.gw-openscience.org/O3/)
- Detector: H1s
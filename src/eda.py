import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch

DATA_DIR = r"D:\GRAVISCOPE_AI\data\processed"
OUTPUT_DIR = r"D:\GRAVISCOPE_AI\outputs"

def plot_waveform(data, name):
    plt.figure(figsize=(12, 4))
    plt.plot(data, linewidth=0.6)
    plt.title(f"Strain Time-Series: {name}")
    plt.xlabel("Time Samples")
    plt.ylabel("Normalized Strain")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_waveform.png"))
    plt.close()

def plot_spectrogram(data, fs, name):
    f, t, Sxx = spectrogram(data, fs=fs, nperseg=2048)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='auto')
    plt.colorbar(label='Power (dB)')
    plt.title(f"Spectrogram: {name}")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_spectrogram.png"))
    plt.close()

def plot_psd(data, fs, name):
    f, Pxx = welch(data, fs=fs, nperseg=4096)
    plt.figure(figsize=(10, 4))
    plt.semilogy(f, Pxx)
    plt.title(f"Power Spectral Density: {name}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [strain²/Hz]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_psd.png"))
    plt.close()

def run_visualization():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for file in os.listdir(DATA_DIR):
        if file.endswith(".npy"):
            path = os.path.join(DATA_DIR, file)
            strain = np.load(path)
            name = file.replace(".npy", "")
            print(f"Visualizing: {file}")
            plot_waveform(strain, name)
            plot_spectrogram(strain, fs=4096, name=name)
            plot_psd(strain, fs=4096, name=name)

if __name__ == "__main__":
    run_visualization()

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Config
DATA_DIR = "D:/GRAVISCOPE_AI/data/processed"
SAVE_DIR = "D:/GRAVISCOPE_AI/outputs/visuals"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_waveform(file_path, save_path):
    with h5py.File(file_path, 'r') as f:
        strain = np.array(f['strain'])
        label = f.attrs.get('label', 'unknown')
        
    plt.figure(figsize=(12, 4))
    plt.plot(strain, linewidth=1)
    plt.title(f"Gravitational Waveform - Label: {label}")
    plt.xlabel("Time Steps")
    plt.ylabel("Strain Amplitude")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Run for each file
for file in os.listdir(DATA_DIR):
    if file.endswith(".hdf5"):
        file_path = os.path.join(DATA_DIR, file)
        save_path = os.path.join(SAVE_DIR, f"{file.replace('.hdf5', '')}.png")
        plot_waveform(file_path, save_path)
        print(f"✅ Saved waveform: {save_path}")

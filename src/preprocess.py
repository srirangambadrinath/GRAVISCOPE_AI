import os
import numpy as np
import h5py

RAW_DIR = r"D:\GRAVISCOPE_AI\data\raw"
OUT_DIR = r"D:\GRAVISCOPE_AI\data\processed"

def extract_strain_from_hdf5(filepath):
    with h5py.File(filepath, 'r') as f:
        strain = f['strain']['Strain'][:]
    return strain

def normalize_signal(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / std

def process_all_files():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for file in os.listdir(RAW_DIR):
        if file.endswith(".hdf5"):
            path = os.path.join(RAW_DIR, file)
            print(f"Processing: {file}")
            strain = extract_strain_from_hdf5(path)
            norm = normalize_signal(strain)
            out_path = os.path.join(OUT_DIR, file.replace(".hdf5", ".npy"))
            np.save(out_path, norm)

if __name__ == "__main__":
    process_all_files()

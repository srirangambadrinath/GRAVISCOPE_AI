import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

from graviscope_model import GraviscopeModel
from graviscope_dataset import GraviscopeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = GraviscopeModel(input_length=4096).to(device)
model.load_state_dict(torch.load("D:/GRAVISCOPE_AI/models/graviscope_trained.pt", map_location=device), strict=False)
model.eval()

# Dataset
data_dir = "D:/GRAVISCOPE_AI/data/processed"
dataset = GraviscopeDataset(data_dir)

# Output dir
out_dir = "D:/GRAVISCOPE_AI/outputs/attention"
os.makedirs(out_dir, exist_ok=True)

# Visualize attention for a few samples
for i in range(min(5, len(dataset))):
    waveform, label = dataset[i]
    input_tensor = waveform.unsqueeze(0).to(device)  # (1, seq_len)

    with torch.no_grad():
        logits, attn_weights = model.forward_with_attention(input_tensor)

    attn_weights = attn_weights.squeeze().cpu().numpy()
    waveform = waveform.cpu().numpy()

    # Interpolate attention to match input length
    attn_resized = np.interp(np.arange(len(waveform)), 
                             np.linspace(0, len(waveform), len(attn_weights)), 
                             attn_weights)

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(waveform, label='Strain')
    plt.fill_between(np.arange(len(waveform)), 0, attn_resized * max(waveform), 
                     color='red', alpha=0.4, label='Attention')
    plt.title(f"Waveform with Attention Map (Label: {label})")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude / Attention")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"attention_{i}.png"))
    plt.close()
    print(f"✅ Saved: attention_{i}.png")

# visualize_attention.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

from graviscope_model import GraviscopeModel  # Use exact spelling
from graviscope_dataset import GraviscopeDataset

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == 'cuda':
    print("  -> Name:", torch.cuda.get_device_name(0))

# Load model
model = GraviscopeModel(input_length=4096).to(device)
model_path = "D:/GRAVISCOPE_AI/models/graviscope_trained.pt"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

# Ensure the model has forward_with_attention
if not hasattr(model, 'forward_with_attention'):
    def forward_with_attention(self, x):
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.cnn(x)     # (batch, cnn_out, seq_len/4)
        x = x.permute(0, 2, 1)  # (batch, seq_len/4, cnn_out)
        gru_out, _ = self.gru(x)
        weights = torch.softmax(self.attention.attn(gru_out), dim=1)
        attn_out = torch.sum(weights * gru_out, dim=1)
        logits = self.classifier(attn_out)
        return logits, weights.squeeze(-1)

    import types
    model.forward_with_attention = types.MethodType(forward_with_attention, model)

# Dataset
data_dir = "D:/GRAVISCOPE_AI/data/processed"
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"❌ Data folder not found: {data_dir}")

dataset = GraviscopeDataset(data_dir)

if len(dataset) == 0:
    raise ValueError("❌ Dataset is empty!")

# Output dir
out_dir = "D:/GRAVISCOPE_AI/outputs/attention"
os.makedirs(out_dir, exist_ok=True)

# Visualize attention
for i in range(min(5, len(dataset))):
    print(f"🔍 Processing sample {i+1}")
    waveform, label = dataset[i]
    input_tensor = waveform.unsqueeze(0).to(device)  # (1, seq_len)

    with torch.no_grad():
        logits, attn_weights = model.forward_with_attention(input_tensor)

    attn_weights = attn_weights.squeeze().cpu().numpy()
    waveform = waveform.cpu().numpy()

    if attn_weights.shape[0] < 10:
        print("⚠️ Warning: attention vector too short — skipping.")
        continue

    # Resize attention map to match waveform
    attn_resized = np.interp(
        np.arange(len(waveform)),
        np.linspace(0, len(waveform), len(attn_weights)),
        attn_weights
    )

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(waveform, label='Strain', color='black')
    plt.fill_between(np.arange(len(waveform)), 0, attn_resized * np.max(np.abs(waveform)), 
                     color='red', alpha=0.4, label='Attention')
    plt.title(f"Waveform with Attention Map (Label: {label})")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude / Attention")
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(out_dir, f"attention_{i}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved: {output_path}")

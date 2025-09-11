import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from graviscope_model import GraviscopeModel
from graviscope_dataset import GraviscopeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == 'cuda':
    print("  -> Name:", torch.cuda.get_device_name(0))

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
data_dir = "D:/GRAVISCOPE_AI/data/processed"

# Auto-generate dummy HDF5 data if missing
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not any(f.endswith(".hdf5") for f in os.listdir(data_dir)):
    import h5py
    import numpy as np
    for i in range(5):
        file_path = os.path.join(data_dir, f"sample_{i}.hdf5")
        with h5py.File(file_path, 'w') as f:
            strain = np.random.normal(0, 1, size=(4096,))
            f.create_dataset('strain', data=strain)
            f.attrs['label'] = int(i % 2)
    print("✅ Dummy HDF5 files generated.")

train_dataset = GraviscopeDataset(data_dir=data_dir, mode='train')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

sample_input_length = 4096
model = GraviscopeModel(input_length=sample_input_length).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        print(f"\nEpoch [{epoch+1}] complete. Avg Loss: {total_loss/len(train_loader):.4f}\n")

    # Save trained model
    torch.save(model.state_dict(), "D:/GRAVISCOPE_AI/models/graviscope_trained.pt")
    print("✅ Model saved.")

if __name__ == '__main__':
    train()

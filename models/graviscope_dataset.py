import os
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class GraviscopeDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.data_dir = os.path.abspath(data_dir)
        self.mode = mode

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Path does not exist: {self.data_dir}")

        self.file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.hdf5')]
        if len(self.file_list) == 0:
            raise RuntimeError(f"No .hdf5 files found in {self.data_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        with h5py.File(file_path, 'r') as f:
            data = np.array(f['strain'])
            label = f.attrs.get('label', 0)

        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return data, label

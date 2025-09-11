import os
import h5py

data_dir = "D:/GRAVISCOPE_AI/data/processed"

if not os.path.exists(data_dir):
    print("❌ Folder does not exist.")
    exit()

files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
if not files:
    print("❌ No .hdf5 files found.")
    exit()

print(f"✅ Found {len(files)} .hdf5 files\n")

for f_name in files:
    print(f"🔍 Checking file: {f_name}")
    with h5py.File(os.path.join(data_dir, f_name), 'r') as f:
        if 'strain' in f:
            print(f"  ✅ 'strain' dataset found → shape: {f['strain'].shape}")
        else:
            print(f"  ❌ Missing 'strain' dataset")

        label = f.attrs.get('label', None)
        print(f"  🏷  Label: {label if label is not None else '❌ Not found'}\n")

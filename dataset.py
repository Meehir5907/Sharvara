import os
import torch
from torch.utils.data import Dataset, DataLoader

class FlowDataset(Dataset):
    def __init__(self, tensor_dir="tensors"):
        self.tensor_dir = tensor_dir
        # Note: os.listdir on millions of files will take a few moments upon initialization
        self.file_list = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.tensor_dir, self.file_list[idx])
        
        # Load the dictionary containing both the tensor and the stats
        data = torch.load(file_path, weights_only=True)
        
        flow_tensor = data['tensor']  # Expected Shape: (10, 60)
        stats = data['stats']         # Dict containing 'label', 'duration', etc.
        
        raw_label = str(stats['label'])
        
        # 3-Class Classification Logic
        if raw_label.startswith('From-Botnet'):
            encoded_label = 2  # Malicious Botnet Traffic
        elif raw_label.startswith('From-Normal'):
            encoded_label = 1  # Verified Benign Human Traffic
        else:
            encoded_label = 0  # Unverified Background Noise
            
        return flow_tensor, encoded_label

# Quick execution test
if __name__ == "__main__":
    print("[INFO] Initializing Dataset...")
    dataset = FlowDataset()
    print(f"[SUCCESS] Dataset mapped with {len(dataset)} flows.")
    
    if len(dataset) > 0:
        sample_tensor, sample_label = dataset[0]
        print(f"Tensor Shape: {sample_tensor.shape}")
        
        # Map back to string for verification in the printout
        class_names = {0: "Background", 1: "Normal", 2: "Botnet"}
        print(f"Encoded Flow Label: {sample_label} ({class_names[sample_label]})")

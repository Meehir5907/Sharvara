import os
import torch
from torch.utils.data import Dataset, DataLoader

class FlowDataset(Dataset):
    def __init__(self, tensor_dir="tensors"):
        self.tensor_dir = tensor_dir
        self.file_paths = [
            os.path.join(tensor_dir, f) 
            for f in os.listdir(tensor_dir) 
            if f.endswith('.pt')
        ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        payload = torch.load(path, weights_only=False)
        
        tensor = payload['tensor']
        label = str(payload['stats']['label'])
        
        if "Botnet" in label:
            cls_idx = 2
        elif "Normal" in label:
            cls_idx = 1
        else:
            cls_idx = 0
            
        return tensor, torch.tensor(cls_idx, dtype=torch.long)

if __name__ == "__main__":
    print("[INFO] Initializing Dataset...")
    dataset = FlowDataset()
    print(f"[SUCCESS] Dataset mapped with {len(dataset)} flows.")
    
    if len(dataset) > 0:
        sample_tensor, sample_label = dataset[0]
        print(f"Tensor Shape: {sample_tensor.shape}")
        
        class_names = {0: "Background", 1: "Normal", 2: "Botnet"}
        print(f"Encoded Flow Label: {sample_label} ({class_names[sample_label.item()]})")

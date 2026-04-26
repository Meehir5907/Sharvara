import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import FlowDataset
from model import NChannelTransformer

def evaluate_model():
    fraction = 1.0
    if len(sys.argv) > 1:
        try:
            fraction = float(sys.argv[1])
        except ValueError:
            print("[ERROR] Argument must be a float (e.g., 0.10)")
            sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Execution Device: {device}")

    print("[INFO] Mapping directory and recreating split...")
    full_dataset = FlowDataset(tensor_dir="tensors")
    total_files = len(full_dataset)
    subset_size = int(total_files * fraction)
    
    if fraction < 1.0:
        working_dataset, _ = random_split(
            full_dataset, 
            [subset_size, total_files - subset_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        working_dataset = full_dataset

    train_size = int(0.8 * subset_size)
    val_size = int(0.1 * subset_size)
    test_size = subset_size - train_size - val_size
    
    _, _, test_ds = random_split(
        working_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # Must match train.py exactly
    )

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    model_path = os.path.join("outputs", "n_channel_model.pth")
    if not os.path.exists(model_path):
        model_path = "n_channel_model.pth"

    print(f"[INFO] Loading weights from {model_path}...")
    model = NChannelTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    num_classes = 3
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    print("\n[INFO] Generating Confusion Matrix on Test Set...")
    with torch.no_grad():
        for batch_tensors, batch_labels in tqdm(test_loader, desc="Evaluating", leave=False):
            batch_tensors = batch_tensors.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_tensors)
            _, predicted = torch.max(outputs, 1)
            
            for t, p in zip(batch_labels.view(-1), predicted.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

    classes = ["Background", "Normal", "Botnet"]
    
    print("\n" + "="*50)
    print(" "*15 + "CONFUSION MATRIX")
    print("="*50)
    print(f"{'':<15} | {'Predicted':^30}")
    print(f"{'True Label':<15} | {classes[0]:<10} {classes[1]:<10} {classes[2]:<10}")
    print("-" * 50)
    
    for i in range(num_classes):
        row_str = f"{classes[i]:<15} | "
        for j in range(num_classes):
            row_str += f"{conf_matrix[i, j].item():<10} "
        print(row_str)
    
    print("="*50 + "\n")

    for i in range(num_classes):
        total_true = conf_matrix[i, :].sum().item()
        correct = conf_matrix[i, i].item()
        accuracy = (correct / total_true * 100) if total_true > 0 else 0
        print(f"-> {classes[i]} Accuracy: {accuracy:.2f}% ({correct}/{total_true})")

if __name__ == "__main__":
    evaluate_model()

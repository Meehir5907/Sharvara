import os
import sys
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

from dataset import FlowDataset
from model import NChannelTransformer

def evaluate_model(model, dataloader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_tensors, batch_labels in tqdm(dataloader, desc=desc, leave=False):
            batch_tensors = batch_tensors.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_tensors)
            loss = criterion(outputs, batch_labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
    avg_loss = total_loss / len(dataloader)
    accuracy = (correct / total_samples) * 100
    return avg_loss, accuracy

def get_dynamic_weights(dataset, num_classes=3, cache_file="outputs/class_weights.json"):
    if os.path.exists(cache_file):
        print(f"[INFO] Loading cached class weights from {cache_file}...")
        with open(cache_file, 'r') as f:
            weights = json.load(f)
        return torch.tensor(weights, dtype=torch.float32)
        
    print("[INFO] Auto-detecting class distribution. This will take a few seconds...")
    class_counts = [0] * num_classes
    
    for i in tqdm(range(len(dataset)), desc="Counting Labels", leave=False):
        _, label = dataset[i]
        label_idx = int(label.item() if torch.is_tensor(label) else label)
        class_counts[label_idx] += 1
        
    print(f"[INFO] Detected Counts -> BG: {class_counts[0]} | Normal: {class_counts[1]} | Botnet: {class_counts[2]}")
    
    total_samples = sum(class_counts)
    weights = []
    for count in class_counts:
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total_samples / (num_classes * count))
            
    print(f"[INFO] Calculated Weights: {[round(w, 4) for w in weights]}")
    
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(weights, f)
        
    return torch.tensor(weights, dtype=torch.float32)

def train_model():
    fraction = 1.0
    if len(sys.argv) > 1:
        try:
            fraction = float(sys.argv[1])
            if not (0.0 < fraction <= 1.0):
                raise ValueError
        except ValueError:
            print("[ERROR] Argument must be a float between 0.0 and 1.0 (e.g., 0.1 for 10%)")
            sys.exit(1)

    epochs = 5
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Execution Device: {device}")

    print(f"[INFO] Mapping directory... (This may take a moment for millions of files)")
    full_dataset = FlowDataset(tensor_dir="tensors")
    total_files = len(full_dataset)
    
    subset_size = int(total_files * fraction)
    print(f"[INFO] Using {fraction*100}% of the data: {subset_size} / {total_files} tensors.")
    
    if fraction < 1.0:
        generator = torch.Generator().manual_seed(42)
        working_dataset, _ = random_split(
            full_dataset, 
            [subset_size, total_files - subset_size],
            generator=generator
        )
    else:
        working_dataset = full_dataset

    train_size = int(0.8 * subset_size)
    val_size = int(0.1 * subset_size)
    test_size = subset_size - train_size - val_size
    
    print(f"[INFO] Data Split -> Train: {train_size} | Val: {val_size} | Test: {test_size}")
    
    train_ds, val_ds, test_ds = random_split(
        working_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8)

    model = NChannelTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    weights = get_dynamic_weights(working_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    csv_filename = os.path.join(out_dir, "training_results.csv")
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train_Loss", "Train_Accuracy(%)", "Val_Loss", "Val_Accuracy(%)"])

        print("\n[INFO] Commencing Training Loop...")
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            correct_train = 0
            total_train_samples = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch_tensors, batch_labels in pbar:
                batch_tensors = batch_tensors.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = model(batch_tensors)
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train_samples += batch_labels.size(0)
                correct_train += (predicted == batch_labels).sum().item()
                
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / len(train_loader)
            train_acc = (correct_train / total_train_samples) * 100
            
            avg_val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, desc=f"Epoch {epoch+1} [Val]")
            
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
            writer.writerow([epoch + 1, f"{avg_train_loss:.4f}", f"{train_acc:.2f}", f"{avg_val_loss:.4f}", f"{val_acc:.2f}"])

        print("\n[INFO] Training Complete. Evaluating on unseen Test Set...")
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, desc="Testing")
        print(f"-> FINAL TEST RESULTS | Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%\n")
        
        writer.writerow([])
        writer.writerow(["FINAL TEST RESULTS", f"Loss: {test_loss:.4f}", f"Accuracy: {test_acc:.2f}%", "", ""])

    model_path = os.path.join(out_dir, "n_channel_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[SUCCESS] Model saved to '{model_path}'")
    print(f"[SUCCESS] Metrics saved to '{csv_filename}'")

if __name__ == "__main__":
    train_model()

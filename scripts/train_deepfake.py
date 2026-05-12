import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.data.sequence_dataset import DeepfakeSequenceDataset
from src.models.efficientnet_gru import DeepfakeDetector
import numpy as np

def train():
    # 1. SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Paths
    faces_dir = "data/faces"
    save_dir = "models/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 2. HYPERPARAMETERS
    batch_size = 8 # Sequence models are VRAM hungry! 8 sequences * 16 frames = 128 images per batch
    learning_rate = 1e-4
    epochs = 30
    best_val_loss = float('inf')

    # 3. DATA LOADERS
    full_dataset = DeepfakeSequenceDataset(faces_dir, is_train=True)
    if len(full_dataset) == 0:
        print("ERROR: No extracted sequences found! Please run `preprocess_dataset.py` first.")
        return

    # 80/20 train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Loaded {len(train_ds)} train sequences and {len(val_ds)} validation sequences.")

    # 4. MODEL, LOSS, OPTIMIZER
    model = DeepfakeDetector(cnn_backbone='efficientnet_b0').to(device)
    
    # We are classifying 0 (real) vs 1 (fake)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 5. TRAINING LOOP
    print(f"Starting Deepfake Sequence Training...")
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(device) # Shape: [Batch, 16, 3, 224, 224]
            labels = labels.to(device)       # Shape: [Batch]

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print("--------------------------\n")

        # 6. SAVE BEST MODEL
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{save_dir}/best_deepfake_detector.pth")
            print(f">> New Best Validation Loss: {best_val_loss:.4f}")

    print("Sequence Training Complete!")

if __name__ == "__main__":
    train()

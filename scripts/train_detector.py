import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.wider_dataset import WiderFaceDataset
from src.models.pretrained_detector import PretrainedDetector
from src.models.losses import IoULoss
import cv2
import numpy as np

def calc_metrics(pred, target):
    with torch.no_grad():
        # Coordinates
        p_x2, p_y2 = pred[:, 0] + pred[:, 2], pred[:, 1] + pred[:, 3]
        t_x2, t_y2 = target[:, 0] + target[:, 2], target[:, 1] + target[:, 3]

        # Intersection
        i_x1 = torch.max(pred[:, 0], target[:, 0])
        i_y1 = torch.max(pred[:, 1], target[:, 1])
        i_x2 = torch.min(p_x2, t_x2)
        i_y2 = torch.min(p_y2, t_y2)

        i_w = torch.clamp(i_x2 - i_x1, min=0)
        i_h = torch.clamp(i_y2 - i_y1, min=0)
        inter_area = i_w * i_h

        # Union
        union_area = (pred[:, 2]*pred[:, 3]) + (target[:, 2]*target[:, 3]) - inter_area + 1e-7
        iou = inter_area / union_area
        
        acc_50 = (iou > 0.5).float().mean()
        l1_error = torch.nn.functional.l1_loss(pred, target) * 224 # convert back to pixel average error
        
        return iou.mean().item(), acc_50.item(), l1_error.item()

def train():
    # 1. SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Paths
    root_dir = "data/cnn_faces/WIDER_train/images"
    split_file = "data/cnn_faces/wider_face_split/wider_face_train_bbx_gt.txt"
    save_dir = "models/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("reports/detector_samples", exist_ok=True)

    # 2. HYPERPARAMETERS (Optimized for Deep Scratch)
    batch_size = 256 # Optimized for ~7GB VRAM
    learning_rate = 3e-4 # Golden rule learning rate for transfer learning
    epochs = 150
    best_iou_loss = float('inf')

    # 3. DATA LOADERS (With Augmentation)
    train_ds = WiderFaceDataset(root_dir, split_file, is_train=True)
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    # 4. MODEL, LOSS, OPTIMIZER
    model = PretrainedDetector(freeze_backbone=False).to(device)
    
    # Custom IoU Loss: This is the magic for 90%+
    criterion = IoULoss() 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # OneCycleLR: Starts slow, gets fast, then cools down. Best for scratch training.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs
    )

    # 5. TRAINING LOOP
    print(f"Starting Fine-Tuning Training (ResNet18 + IoU)...")
    for epoch in range(epochs):
        model.train()
        total_loss, total_iou, total_acc, total_l1 = 0, 0, 0, 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Step the OneCycle scheduler every BATCH
            scheduler.step()

            total_loss += loss.item()
            
            # Metrics
            iou, acc, l1 = calc_metrics(outputs, targets)
            total_iou += iou
            total_acc += acc
            total_l1 += l1

            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}, IoU: {iou*100:.1f}%, Acc@50: {acc*100:.1f}%, L1_Err: {l1:.1f}px")

        avg_loss = total_loss / len(train_loader)
        avg_iou = total_iou / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        avg_l1 = total_l1 / len(train_loader)
        
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Loss:       {avg_loss:.4f}")
        print(f"Mean IoU:   {avg_iou*100:.1f}%")
        print(f"Acc (IoU>.5): {avg_acc*100:.1f}%")
        print(f"Pixel Err:  {avg_l1:.1f} px")
        print("--------------------------\n")

        # 6. VISUALIZATION (Sample Grid)
        model.eval()
        with torch.no_grad():
            grid_img = []
            for j in range(5):
                idx = np.random.randint(0, len(train_ds))
                img_t, target_t = train_ds[idx]
                img_in = img_t.unsqueeze(0).to(device)
                pred_t = model(img_in).squeeze(0).cpu().numpy()
                
                # Unnormalize image for display
                img_np = img_t.cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                img_np = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                tx, ty, tw, th = target_t.numpy() * 224
                px, py, pw, ph = pred_t * 224
                
                cv2.rectangle(img_np, (int(tx), int(ty)), (int(tx+tw), int(ty+th)), (0, 255, 0), 2)
                cv2.rectangle(img_np, (int(px), int(py)), (int(px+pw), int(py+ph)), (0, 0, 255), 2)
                grid_img.append(img_np)
            
            final_grid = np.hstack(grid_img)
            cv2.imwrite(f"reports/detector_samples/epoch_{epoch+1}.png", final_grid)

        # 7. SAVE BEST MODEL
        if avg_loss < best_iou_loss:
            best_iou_loss = avg_loss
            torch.save(model.state_dict(), f"{save_dir}/best_detector.pth")
            print(f">> New Best IoU Loss: {best_iou_loss:.6f}")
        
        torch.save(model.state_dict(), f"{save_dir}/detector_last.pth")

    print("Fine-Tuning Training Complete!")

if __name__ == "__main__":
    train()

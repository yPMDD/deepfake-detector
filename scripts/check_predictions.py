import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.data.wider_dataset import WiderFaceDataset
from src.models.pretrained_detector import PretrainedDetector
import numpy as np
import os

def check_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    root_dir = "data/cnn_faces/WIDER_train/images"
    split_file = "data/cnn_faces/wider_face_split/wider_face_train_bbx_gt.txt"
    checkpoint = "models/checkpoints/best_detector.pth"
    
    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint {checkpoint} not found!")
        return

    # Load Dataset
    ds = WiderFaceDataset(root_dir, split_file)
    
    # Load Model
    model = PretrainedDetector(freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    print(f"--- Predicting on 10 Random Samples ---")
    
    with torch.no_grad():
        for i in range(10):
            # Take a random sample to see variety
            idx = np.random.randint(0, len(ds))
            img_t, target_t = ds[idx]
            img_in = img_t.unsqueeze(0).to(device)
            
            output = model(img_in).squeeze(0).cpu().numpy()
            target = target_t.numpy()
            
            print(f"Sample {idx}:")
            print(f"  TRUTH: {target}")
            print(f"  GUESS: {output}")
            print(f"  DIFF : {np.abs(target - output)}")
            print("-" * 20)

if __name__ == "__main__":
    check_predictions()

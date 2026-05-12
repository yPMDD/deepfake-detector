import os
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WiderFaceDataset(Dataset):
    """
    Enhanced WIDER Face Dataset with Data Augmentation and Bounding Box handling.
    """
    def __init__(self, root_dir, split_file, image_size=224, min_face_size=50, is_train=True):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.samples = []

        # Define Augmentation Pipeline
        if is_train:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Annotation file not found: {split_file}")

        print(f"Parsing WIDER annotations from {split_file}...")
        with open(split_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            img_path = lines[i].strip()
            num_faces = int(lines[i+1].strip())
            i += 2
            
            max_area = 0
            best_bbox = None
            
            for _ in range(num_faces):
                parts = list(map(int, lines[i].strip().split()))
                x, y, w, h = parts[0], parts[1], parts[2], parts[3]
                invalid = parts[7]
                
                if invalid == 0 and w >= min_face_size and h >= min_face_size:
                    area = w * h
                    if area > max_area:
                        max_area = area
                        best_bbox = [x, y, w, h]
                i += 1
            
            if num_faces == 0: i += 1 

            if best_bbox:
                self.samples.append({"img_path": img_path, "bbox": best_bbox})

        print(f"Loaded {len(self.samples)} valid images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.root_dir / sample["img_path"]
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Bbox in COCO format: [x, y, w, h]
        bbox = sample["bbox"]
        
        # Apply Augmentations
        transformed = self.transform(image=img, bboxes=[bbox], class_labels=[0])
        img_tensor = transformed['image']
        
        # Get transformed bbox (COCO format)
        if len(transformed['bboxes']) > 0:
            new_bbox = transformed['bboxes'][0]
            # Normalize to 0-1 relative to the 224x224 output
            nx = new_bbox[0] / self.image_size
            ny = new_bbox[1] / self.image_size
            nw = new_bbox[2] / self.image_size
            nh = new_bbox[3] / self.image_size
        else:
            # Fallback if augmentation pushed face out (rare)
            nx, ny, nw, nh = 0.0, 0.0, 0.0, 0.0
            
        bbox_tensor = torch.tensor([nx, ny, nw, nh], dtype=torch.float32)
        
        return img_tensor, bbox_tensor

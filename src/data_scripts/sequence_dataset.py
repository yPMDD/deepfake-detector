import os
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepfakeSequenceDataset(Dataset):
    """
    Loads 16-frame sequences from extracted face crops.
    Returns: tensor shape [16, 3, 224, 224] and label (0: real, 1: fake)
    """
    def __init__(self, faces_dir, is_train=True, frames_per_video=16):
        self.faces_dir = Path(faces_dir)
        self.frames_per_video = frames_per_video
        self.is_train = is_train
        
        self.samples = []
        
        # Collect video ids
        for label, class_name in enumerate(["real", "fake"]):
            class_dir = self.faces_dir / class_name
            if not class_dir.exists():
                continue
                
            # Group by video id
            # files are like videoID_frame_0.png
            files = list(class_dir.glob("*.png"))
            video_frames = {}
            for f in files:
                vid_id = f.stem.rsplit('_frame_', 1)[0]
                if vid_id not in video_frames:
                    video_frames[vid_id] = []
                video_frames[vid_id].append(f)
            
            # Filter for complete sequences
            for vid_id, frame_files in video_frames.items():
                if len(frame_files) >= frames_per_video:
                    # Sort by frame index
                    frame_files.sort(key=lambda x: int(x.stem.rsplit('_frame_', 1)[1]))
                    self.samples.append({
                        "frames": frame_files[:frames_per_video],
                        "label": label
                    })

        # Augmentation for frames
        if self.is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets={f'image{i}': 'image' for i in range(1, frames_per_video)})
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets={f'image{i}': 'image' for i in range(1, frames_per_video)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_paths = sample["frames"]
        label = sample["label"]
        
        # Load all frames
        frames = []
        for path in frame_paths:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            
        # Apply the exact same augmentation to ALL frames in the sequence
        # to preserve temporal consistency
        aug_args = {"image": frames[0]}
        for i in range(1, len(frames)):
            aug_args[f"image{i}"] = frames[i]
            
        augmented = self.transform(**aug_args)
        
        # Collect back into a single sequence tensor [T, C, H, W]
        seq_tensors = [augmented["image"]]
        for i in range(1, len(frames)):
            seq_tensors.append(augmented[f"image{i}"])
            
        sequence_tensor = torch.stack(seq_tensors)
        
        return sequence_tensor, torch.tensor(label, dtype=torch.long)

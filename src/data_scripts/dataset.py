import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepfakeVideoDataset(Dataset):
    """
    Dataset class for Deepfake Detection.
    Loads 16 frames per video and applies consistent augmentations.
    """
    def __init__(self, root_dir, transform=None, n_frames=16):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.n_frames = n_frames
        self.samples = [] # List of (video_id, label_tensor, frames_paths)

        # Labels: Real = 0, Fake = 1
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            dir_path = self.root_dir / label_name
            if not dir_path.exists():
                continue

            # Group frames by video ID
            # Filename format: {video_id}_frame_{i}.png
            all_files = sorted(list(dir_path.glob("*.png")))
            video_groups = {}
            for f in all_files:
                video_id = f.name.split("_frame_")[0]
                if video_id not in video_groups:
                    video_groups[video_id] = []
                video_groups[video_id].append(f)

            # Only keep videos that have successfully extracted all frames
            for video_id, frames in video_groups.items():
                if len(frames) == self.n_frames:
                    self.samples.append({
                        "video_id": video_id,
                        "label": label_idx,
                        "frames": sorted(frames, key=lambda x: int(x.name.split("_frame_")[1].split(".")[0]))
                    })

        print(f"Dataset initialized with {len(self.samples)} videos.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]
        frame_paths = sample["frames"]

        frames = []
        for p in frame_paths:
            img = np.array(Image.open(p).convert("RGB"))
            frames.append(img)

        if self.transform:
            # Consistent augmentation across sequence
            # We use additional_targets to apply identical transforms to all frames
            target_keys = {f"image{i}": "image" for i in range(1, len(frames))}
            aug_transform = A.Compose(self.transform.transforms, additional_targets=target_keys)
            
            aug_input = {"image": frames[0]}
            for i in range(1, len(frames)):
                aug_input[f"image{i}"] = frames[i]
            
            augmented = aug_transform(**aug_input)
            
            # Reconstruct the sequence
            frames = [augmented["image"]]
            for i in range(1, len(frames_paths)):
                frames.append(augmented[f"image{i}"])

        # Stack into (T, C, H, W)
        # Note: If ToTensorV2 was in transform, frames are already tensors
        if isinstance(frames[0], np.ndarray):
            # Fallback if no ToTensorV2
            frames = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]
            
        video_tensor = torch.stack(frames) # Shape: (16, 3, 224, 224)
        
        return video_tensor, torch.tensor(label, dtype=torch.long)

def get_transforms(image_size=224, train=True):
    """
    Returns the augmentation pipeline.
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

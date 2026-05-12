import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.efficientnet_gru import DeepfakeDetector
from src.data.preprocessing import FaceExtractor

def predict(video_path, model_path):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at '{video_path}'")
        return

    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights not found at '{model_path}'")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing on: {device}")

    # 1. Initialize Face Extractor (MTCNN)
    print("⏳ Extracting faces from video...")
    extractor = FaceExtractor(image_size=224, margin=20, device=device)
    frames = extractor.sample_frames(video_path, n_frames=16)
    
    if len(frames) < 16:
        print(f"❌ Error: Could not extract 16 frames from the video. Only got {len(frames)}.")
        return

    # 2. Extract bounding boxes/faces (returns list of PIL Images)
    crops = extractor.extract_faces(frames)
    if len(crops) < 16:
        print(f"❌ Error: Could not find faces in all 16 frames. Only found {len(crops)}.")
        return

    # 3. Preprocessing Pipeline (Exact same as validation dataset)
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={f'image{i}': 'image' for i in range(1, 16)})

    # Convert PIL to Numpy arrays
    np_crops = [np.array(crop) for crop in crops]
    
    aug_args = {"image": np_crops[0]}
    for i in range(1, 16):
        aug_args[f"image{i}"] = np_crops[i]
        
    augmented = transform(**aug_args)
    
    # Stack into sequence tensor [1, 16, 3, 224, 224]
    seq_tensors = [augmented["image"]]
    for i in range(1, 16):
        seq_tensors.append(augmented[f"image{i}"])
        
    sequence_tensor = torch.stack(seq_tensors).unsqueeze(0).to(device)

    # 4. Load Model
    print("🧠 Loading AI model...")
    model = DeepfakeDetector(cnn_backbone='efficientnet_b0').to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 5. Inference
    print("🔍 Analyzing sequence...")
    with torch.no_grad():
        logits = model(sequence_tensor)
        probs = F.softmax(logits, dim=1)
        
        # 0 = Real, 1 = Fake (Based on sequence_dataset.py mapping)
        fake_prob = probs[0][1].item() * 100
        real_prob = probs[0][0].item() * 100

    # 6. Results
    print("\n" + "="*40)
    print("🎯 FINAL PREDICTION")
    print("="*40)
    
    if fake_prob > 50.0:
        print(f"🚨 VERDICT: FAKE VIDEO")
        print(f"📊 Confidence: {fake_prob:.2f}%")
    else:
        print(f"✅ VERDICT: REAL VIDEO")
        print(f"📊 Confidence: {real_prob:.2f}%")
    print("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Deepfake Video Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video (.mp4)")
    parser.add_argument("--model", type=str, default="models/checkpoints/best_deepfake_detector.pth", help="Path to model weights")
    args = parser.parse_args()

    predict(args.video, args.model)

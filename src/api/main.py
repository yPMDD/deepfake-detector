import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.efficientnet_gru import DeepfakeDetector
from src.data.preprocessing import FaceExtractor

app = FastAPI(title="Deepfake Detector API")

# Allow React frontend to communicate with API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching model and extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
extractor = None

def load_ai():
    global model, extractor
    if model is None:
        print("🧠 Loading AI Model into Memory...")
        model_path = "models/checkpoints/best_deepfake_detector.pth"
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model not found at {model_path}. Did you run train_deepfake.py?")
        
        model = DeepfakeDetector(cnn_backbone='efficientnet_b0').to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    if extractor is None:
        print("⏳ Initializing Face Extractor...")
        extractor = FaceExtractor(image_size=224, margin=20, device=device)

@app.on_event("startup")
async def startup_event():
    load_ai()

@app.post("/api/detect")
async def detect_deepfake(video: UploadFile = File(...)):
    """
    Accepts a video upload, runs it through MTCNN + EfficientNet-GRU, and returns the prediction.
    """
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Only .mp4, .avi, or .mov files are supported")

    # 1. Save uploaded video to temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            shutil.copyfileobj(video.file, tmp_file)
            tmp_video_path = tmp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    try:
        # 2. Extract faces
        frames = extractor.sample_frames(tmp_video_path, n_frames=16)
        if len(frames) < 16:
            raise HTTPException(status_code=400, detail=f"Could not extract 16 frames from the video. Video might be too short.")

        crops = extractor.extract_faces(frames)
        if len(crops) == 0:
            raise HTTPException(status_code=400, detail="Could not detect any faces in the video.")
            
        # Time-stretch sequence if some frames failed face detection (e.g. motion blur)
        if len(crops) < 16:
            original_length = len(crops)
            stretched_crops = []
            for i in range(16):
                # Evenly distribute the available frames across the 16 required slots
                src_index = min(int((i / 15.0) * (original_length - 1)), original_length - 1)
                stretched_crops.append(crops[src_index])
            crops = stretched_crops

        # 3. Preprocess
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={f'image{i}': 'image' for i in range(1, 16)})

        np_crops = [np.array(crop) for crop in crops]
        
        aug_args = {"image": np_crops[0]}
        for i in range(1, 16):
            aug_args[f"image{i}"] = np_crops[i]
            
        augmented = transform(**aug_args)
        
        seq_tensors = [augmented["image"]]
        for i in range(1, 16):
            seq_tensors.append(augmented[f"image{i}"])
            
        sequence_tensor = torch.stack(seq_tensors).unsqueeze(0).to(device)

        # 4. Predict
        with torch.no_grad():
            logits = model(sequence_tensor)
            probs = F.softmax(logits, dim=1)
            
            fake_prob = probs[0][1].item() * 100
            real_prob = probs[0][0].item() * 100

        is_fake = fake_prob > 50.0
        confidence = fake_prob if is_fake else real_prob

        return {
            "status": "success",
            "verdict": "FAKE" if is_fake else "REAL",
            "confidence": round(confidence, 2),
            "probabilities": {
                "real": round(real_prob, 2),
                "fake": round(fake_prob, 2)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

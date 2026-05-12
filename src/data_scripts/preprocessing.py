import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from pathlib import Path
from typing import List, Optional

class FaceExtractor:
    """
    A unified class to handle frame sampling from videos and face extraction via MTCNN.
    """
    def __init__(self, image_size: int = 224, margin: int = 20, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        # mtcnn initialization
        # select_largest=True ensures we get the main subject for deepfake consistency
        # post_process=False because we want raw PIL/numpy for our own preprocessing later
        self.detector = MTCNN(
            image_size=image_size,
            margin=margin,
            post_process=False,
            device=device,
            select_largest=True,
            keep_all=False # Only the main face
        )
        self.image_size = image_size

    def sample_frames(self, video_path: str, n_frames: int = 16) -> List[np.ndarray]:
        """
        Extract n_frames evenly spaced from the video.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []

        # Calculate step to get n frames
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV) to RGB (MTCNN expects RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames

    def extract_faces(self, frames: List[np.ndarray]) -> List[Image.Image]:
        """
        Detect and crop faces from a list of frames.
        Returns a list of PIL images (face crops).
        """
        face_crops = []
        for frame in frames:
            # Convert numpy to PIL
            pil_img = Image.fromarray(frame)
            
            # Use MTCNN to get the cropped face
            # detector(pil_img) returns a tensor if post_process=True
            # here we want the PIL image crop directly or via the bounding boxes
            boxes, probs = self.detector.detect(pil_img)
            
            if boxes is not None and len(boxes) > 0:
                # filter by probability (lenient for glasses/headsets)
                if probs[0] > 0.75:
                    box = boxes[0].astype(int)
                    # Create the crop manually from the box to have more control
                    # box format: [x1, y1, x2, y2]
                    # We ensure the box doesn't exceed image boundaries
                    x1, y1, x2, y2 = box
                    w, h = pil_img.size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    face_crop = pil_img.crop((x1, y1, x2, y2))
                    # Resize to target size for consistency
                    face_crop = face_crop.resize((self.image_size, self.image_size), Image.LANCZOS)
                    face_crops.append(face_crop)
        
        return face_crops

    def process_video(self, video_path: str, output_dir: Path, n_frames: int = 16) -> int:
        """
        Samples, extracts, and saves face crops to output_dir.
        Returns the number of faces successfully saved.
        """
        frames = self.sample_frames(video_path, n_frames)
        if not frames:
            return 0
            
        crops = self.extract_faces(frames)
        
        # We only save if we got the full set (for consistency) or at least most of them
        # Let's be strict: if we don't get all 16, we skip to keep the temporal model clean
        if len(crops) < n_frames:
            return 0
            
        output_dir.mkdir(parents=True, exist_ok=True)
        video_id = Path(video_path).stem
        
        for i, crop in enumerate(crops):
            crop.save(output_dir / f"{video_id}_frame_{i}.png")
            
        return len(crops)

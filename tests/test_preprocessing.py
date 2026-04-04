import pytest
import cv2
import numpy as np
from pathlib import Path
from src.data.preprocessing import FaceExtractor

def test_sampler_returns_correct_number_of_frames():
    # Find one of the downloaded videos for testing
    video_search = list(Path("data/raw").rglob("*.mp4"))
    if not video_search:
        pytest.skip("No raw videos found to test with.")
        
    video_path = str(video_search[0])
    extractor = FaceExtractor()
    frames = extractor.sample_frames(video_path, n_frames=16)
    
    assert len(frames) == 16
    assert isinstance(frames[0], np.ndarray)
    assert frames[0].shape[2] == 3 # RGB

def test_extractor_confirms_face_shape():
    video_search = list(Path("data/raw").rglob("*.mp4"))
    if not video_search:
        pytest.skip("No raw videos found to test with.")
        
    video_path = str(video_search[0])
    extractor = FaceExtractor(image_size=224)
    
    # Process one video
    frames = extractor.sample_frames(video_path, n_frames=1)
    crops = extractor.extract_faces(frames)
    
    if len(crops) > 0:
        assert crops[0].size == (224, 224)

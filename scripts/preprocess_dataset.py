import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to python path so we can import our modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.preprocessing import FaceExtractor

def main():
    # Configuration
    RAW_DIR = Path("data/raw")
    FACES_DIR = Path("data/faces")
    
    # Define paths for real and fake videos based on the FaceForensics structure
    datasets = {
        "real": RAW_DIR / "original_sequences/youtube/c23/videos",
        "fake": RAW_DIR / "manipulated_sequences/Deepfakes/c23/videos"
    }
    
    extractor = FaceExtractor(image_size=224, margin=20)
    
    for label, video_dir in datasets.items():
        if not video_dir.exists():
            print(f"Directory not found: {video_dir}")
            continue
            
        print(f"\n--- Processing {label} videos ---")
        video_files = list(video_dir.glob("*.mp4"))
        
        output_dir = FACES_DIR / label
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        for video_path in tqdm(video_files, desc=f"Extracting {label}"):
            # We process each video and save 16 frames in its own directory
            # but given our model, we might just want to flat save them in directories
            # let's save them as: data/faces/real/{video_id}/frame_0.png
            # Or just data/faces/real/frame_0.png... actually, it's better to keep
            # the video_id in the filename: 062_frame_0.png
            count = extractor.process_video(str(video_path), output_dir, n_frames=16)
            if count == 16:
                success_count += 1
                
        print(f"Successfully processed {success_count}/{len(video_files)} {label} videos.")

if __name__ == "__main__":
    main()

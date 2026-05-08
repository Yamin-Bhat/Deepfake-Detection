import os
import torch
import torchvision.transforms as T
import av
import numpy as np
import timm
import ssl

# Bypass SSL check for model download
ssl._create_default_https_context = ssl._create_unverified_context

# --- 1. SETUP (With Mac Optimization) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal Performance Shaders (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU (This will be slow!)")

# -----------------------------
# Load Feature Extractor (ResNet50)
# -----------------------------
try:
    model = timm.create_model(
        'resnet50', 
        pretrained=True,
        num_classes=0 # Remove classifier, gives us 2048-dim vectors
    ).to(device).eval()
    print("ResNet50 Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Standard ImageNet normalization
norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize = T.Resize((224, 224))

# --- 2. HELPER FUNCTIONS ---

def read_video_pyav(container, indices):
    """Decode specific frames from the video container."""
    frames = []
    container.seek(0)
    start_idx = indices[0]
    end_idx = indices[-1]
    
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_idx:
            break
        if i >= start_idx and i in indices:
            # PyAV returns frames in RGB format
            frames.append(frame.to_rgb().to_ndarray())
            
    if len(frames) == 0:
        return None
    return np.stack(frames)  # (T, H, W, 3)

def sample_frames(total_frames, clip_len=32):
    """Uniformly sample 32 frames from the video."""
    if total_frames <= clip_len:
        # If video is short, take all frames and repeat last one to fill
        indices = np.arange(0, total_frames)
        indices = np.pad(indices, (0, clip_len - len(indices)), mode='edge')
    else:
        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, clip_len).astype(int)
    return indices

@torch.no_grad()
def extract_features(video_path):
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames
        
        if total_frames == 0:
            total_frames = 300 # Fallback
            
        # Sample 32 frames
        indices = sample_frames(total_frames, clip_len=32)
        frames_np = read_video_pyav(container, indices)
        container.close()
        
        if frames_np is None:
            return None
        
        # Preprocess
        frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0 
        frames = resize(frames)
        frames = norm(frames).to(device) # Shape: (32, 3, 224, 224)
        
        # Forward Pass
        feats = model(frames)  # Output: (32, 2048)
        
        return feats.cpu().numpy()  
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

# ================================
# RUN ON YOUR DATASET
# ================================

# Update paths
raw_video_folder = "/Users/Desktop/rPPG/DATASET_FACES"  
output_folder    = "/Users/Desktop/rPPG/marlin_features" 

def process_folder(folder):
    count = 0
    print(f"Scanning: {folder}")
    print(f"Output: {output_folder}")
    
    # Ensure root output folders exist
    os.makedirs(os.path.join(output_folder, "real"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "fake"), exist_ok=True)
    
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                
                # --- AUTO-DETECT CLASS ---
                # Check if "real" or "fake" is in the path
                if "real" in root.lower() or "real" in video_path.lower():
                    label = "real"
                elif "fake" in root.lower() or "fake" in video_path.lower():
                    label = "fake"
                else:
                    print(f"Skipping {file} (Could not determine real/fake from path)")
                    continue
                
                # Define specific output folder
                save_dir = os.path.join(output_folder, label)
                
                # Construct filename: video_name_video.npy
                base_name = os.path.splitext(file)[0]
                output_path = os.path.join(save_dir, base_name + "_video.npy")
                
                # Skip if already exists
                if os.path.exists(output_path):
                    continue
                    
                print(f"[{label.upper()}] Extracting: {file}...", end="\r")
                feats = extract_features(video_path)
                
                if feats is not None:
                    # Final Check: Shape must be (32, 2048)
                    if feats.shape == (32, 2048):
                        np.save(output_path, feats)
                        count += 1
                    else:
                        print(f"\nShape Mismatch {file}: {feats.shape}")
                else:
                    print(f"\nFailed: {file}")
    
    print(f"\n\nProcessing complete. Extracted features for {count} videos.")

if __name__ == "__main__":
    if not os.path.exists(raw_video_folder):
        print(f"Error: Input folder {raw_video_folder} not found.")
    else:
        process_folder(raw_video_folder)
import os
import cv2
import json
import torch
import numpy as np
from face_detector import FacenetDetector, VideoDataset

INPUT_ROOT = "/Users/yaminmohammadbhat/Desktop/rPPG/DATASET"
OUTPUT_ROOT = "/Users/yaminmohammadbhat/Desktop/rPPG/DETECTED_JSON"
CLASSES = ["real", "fake"]


DEVICE = "cpu"

print(f"Running Detection on: {DEVICE}")
# ----------------------------------------

def process_video(video_path, output_json, detector):
    try:
        dataset = VideoDataset([video_path])
        if len(dataset) == 0: 
            print(f"   Skipping (Empty/Unreadable): {os.path.basename(video_path)}")
            return
            
        _, frame_ids, frames = dataset[0]

        if not frames:
            print(f"   Skipping (No frames extracted): {os.path.basename(video_path)}")
            return

        fps_cap = cv2.VideoCapture(video_path)
        fps = fps_cap.get(cv2.CAP_PROP_FPS)
        width = int(fps_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(fps_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cap.release()
        
        if fps < 1 or np.isnan(fps): fps = 30.0

        results = {
            "video_info": {
                "fps": fps,
                "width": width,
                "height": height
            },
            "detections": {}
        }

        batch_size = detector._batch_size
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_ids = frame_ids[i:i + batch_size]

            try:
                batch_boxes = detector._detect_faces(batch_frames)
            except Exception as e:
                print(f"   Batch error: {e}")
                batch_boxes = [None] * len(batch_frames)

            for fid, boxes in zip(batch_ids, batch_boxes):
                if boxes is None:
                    results["detections"][str(fid)] = []
                else:
                    clean = []
                    for (x1, y1, x2, y2) in boxes:
                
                        clean.append([
                            int(x1 * 2),  
                            int(y1 * 2),
                            int(x2 * 2),
                            int(y2 * 2)
                        ])
                    results["detections"][str(fid)] = clean

        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

        
    except Exception as e:
        print(f"Failed to process {os.path.basename(video_path)}: {e}")

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    try:
        detector = FacenetDetector(device=DEVICE)
    except:
        print("MPS/CUDA failed, falling back to CPU...")
        detector = FacenetDetector(device="cpu")

    for cls in CLASSES:
        in_dir = os.path.join(INPUT_ROOT, cls)
        out_dir = os.path.join(OUTPUT_ROOT, cls)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(in_dir):
            print(f"Warning: Folder {in_dir} not found.")
            continue

        files = [f for f in os.listdir(in_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        files = [f for f in files if not f.startswith("._")]
        
        print(f"Found {len(files)} videos in '{cls}'")

        for i, video in enumerate(files):
            video_path = os.path.join(in_dir, video)
            name = os.path.splitext(video)[0]
            output_json = os.path.join(out_dir, f"{name}.json")
            
            if os.path.exists(output_json):
                 continue

            print(f"[{i+1}/{len(files)}] Detecting: {video}...", end='\r')
            process_video(video_path, output_json, detector)

    print("\n\nCompleted detection!.")

if __name__ == "__main__":
    main()
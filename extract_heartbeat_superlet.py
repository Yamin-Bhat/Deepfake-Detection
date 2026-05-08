import os
import cv2
import numpy as np
import scipy.signal
import scipy.ndimage
import sys

import mediapipe as mp
try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh

INPUT_ROOT_DIR = "/Users/yaminmohammadbhat/Desktop/rPPG/DATASET_FACES"
OUTPUT_ROOT_DIR = "/Users/yaminmohammadbhat/Desktop/rPPG/OUTPUT_HEARTBEAT" 
TARGET_SHAPE = (64, 256)

ROI_ZONES = {
    "forehead": [10, 338, 297, 332, 284, 251, 389, 356],
    "left_cheek": [330, 347, 346, 352, 411, 427, 426, 266],
    "right_cheek": [101, 118, 117, 123, 187, 50, 36, 137]
}

MORLET_SD_SPREAD = 6
MORLET_SD_FACTOR = 2.5

def computeWaveletSize(fc, nc, fs):
    if fc <= 0: return 1
    sd = (nc / 2) * (1 / np.abs(fc)) / MORLET_SD_FACTOR
    return int(2 * np.floor(np.round(sd * fs * MORLET_SD_SPREAD) / 2) + 1)

def gausswin(size, alpha):
    halfSize = int(np.floor(size / 2))
    idiv = alpha / halfSize
    t = (np.array(range(size), dtype=np.float64) - halfSize) * idiv
    return np.exp(-(t * t) * 0.5)

def morlet(fc, nc, fs):
    if fc <= 0: return np.zeros(1)
    size = computeWaveletSize(fc, nc, fs)
    half = int(np.floor(size / 2))
    gauss = gausswin(size, MORLET_SD_SPREAD / 2)
    igsum = 1 / gauss.sum()
    ifs = 1 / fs
    t = (np.array(range(size), dtype=np.float64) - half) * ifs
    return gauss * np.exp(2 * np.pi * fc * t * 1j) * igsum

def superlet_transform(signal, fs, foi, base_cycles=2, ord=(2, 10)):
    inputSize = len(signal)
    orders = np.linspace(ord[0], ord[1], len(foi))
    result = np.zeros((len(foi), inputSize), dtype=np.float64)
    
    for i, fc in enumerate(foi):
        nc = orders[i]
        nWavelets = int(np.ceil(nc))
        poolBuffer = np.ones(inputSize, dtype=np.float64)
        
        for j in range(nWavelets):
            cycle = (j + 1) * base_cycles
            wavelet = morlet(fc, cycle, fs)
            
            conv = scipy.signal.fftconvolve(signal, wavelet, mode='same')
            poolBuffer *= (2 * np.abs(conv)**2)
            
        result[i, :] = poolBuffer ** (1.0 / nWavelets)
        
    return result

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def get_roi_mean(frame, landmarks, indices):
    h, w, _ = frame.shape
    points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices], np.int32)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    
    mean_val = cv2.mean(frame[:, :, 1], mask=mask)[0]
    return mean_val

def process_video_3d(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1 or fps > 120 or np.isnan(fps): fps = 30.0
    
    signals = {"forehead": [], "left_cheek": [], "right_cheek": []}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            for zone, indices in ROI_ZONES.items():
                val = get_roi_mean(frame, lms, indices)
                signals[zone].append(val)
        else:
            for zone in signals:
                signals[zone].append(signals[zone][-1] if signals[zone] else 0)

    cap.release()
    
    if len(signals["forehead"]) < 30: return None
    
    channels = []
    foi = np.linspace(0.75, 4.0, TARGET_SHAPE[0]) 
    
    for zone in ["forehead", "left_cheek", "right_cheek"]:
        sig = np.array(signals[zone])
        
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
        
        scalogram = superlet_transform(sig, fps, foi) # Shape (64, Time)
        
        resized = cv2.resize(scalogram, (TARGET_SHAPE[1], TARGET_SHAPE[0]))
        
        mn, mx = np.min(resized), np.max(resized)
        if mx - mn > 0:
            resized = (resized - mn) / (mx - mn)
            
        channels.append(resized)
        
    return np.stack(channels, axis=0).astype(np.float32)

if __name__ == "__main__":
    os.makedirs(os.path.join(OUTPUT_ROOT_DIR, "real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT_DIR, "fake"), exist_ok=True)

    print(f"Scanning: {INPUT_ROOT_DIR}")
    print(f"Output: {OUTPUT_ROOT_DIR}")
    
    count = 0
    for root, _, files in os.walk(INPUT_ROOT_DIR):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                
                if "real" in root.lower() or "real" in video_path.lower():
                    label = "real"
                elif "fake" in root.lower() or "fake" in video_path.lower():
                    label = "fake"
                else:
                    print(f"Skipping {file} (Unknown class)")
                    continue
                
                filename = os.path.splitext(file)[0]
                save_path = os.path.join(OUTPUT_ROOT_DIR, label, filename + "_superlet.npy")
                
                if os.path.exists(save_path): continue
                    
                print(f"[{label.upper()}] 3D Superlet: {filename}...", end='\r')
                
                try:
                    tensor = process_video_3d(video_path)
                    if tensor is not None:
                        if tensor.shape == (3, 64, 256):
                            np.save(save_path, tensor)
                            count += 1
                        else:
                            print(f"\nShape error {filename}: {tensor.shape}")
                except Exception as e:
                    print(f"\nError {filename}: {e}")

    print(f"\n\nDone! Processed {count} videos.")
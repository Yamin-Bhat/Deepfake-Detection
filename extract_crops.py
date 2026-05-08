import cv2
import json
import numpy as np
import os

INPUT_ROOT = "/Users/Desktop/rPPG/DATASET"
DETECTION_ROOT = "/Users/Desktop/rPPG/DETECTED_JSON"
OUTPUT_ROOT = "/Users/Desktop/rPPG/DATASET_FACES"

CLASSES = ["real", "fake"]

FACE_WIDTH = 224
FACE_HEIGHT = 224

IOU_THRESHOLD = 0.3
SMOOTHING_ALPHA = 0.7
MAX_MISSING_FRAMES = 10

MIN_FACE_AREA_RATIO = 0.04   
BLUR_THRESHOLD = 50.0   

def is_blurry(face):
    try:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score < BLUR_THRESHOLD
    except:
        return True

def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (areaA + areaB - inter)


class Track:
    def __init__(self, tid, bbox, fps, base_name, out_dir):
        self.id = tid
        self.bbox = np.array(bbox, dtype=np.float32)
        self.smooth = self.bbox.copy()
        self.missing = 0
        
        self.filepath = os.path.join(out_dir, f"{base_name}_temp_{tid}.avi")

         
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        self.writer = cv2.VideoWriter(
            self.filepath,
            fourcc, fps, (FACE_WIDTH, FACE_HEIGHT)
        )

    def update(self, bbox):
        bbox = np.array(bbox, dtype=np.float32)
        self.smooth = SMOOTHING_ALPHA * self.smooth + (1 - SMOOTHING_ALPHA) * bbox
        self.bbox = bbox
        self.missing = 0

    def miss(self):
        self.missing += 1

    def dead(self):
        return self.missing > MAX_MISSING_FRAMES

    def write(self, frame):
        x1, y1, x2, y2 = self.smooth.astype(int)
        h, w = frame.shape[:2]
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w,x2), min(h,y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return


        try:
            crop = cv2.resize(crop, (FACE_WIDTH, FACE_HEIGHT))
            self.writer.write(crop)
        except Exception as e:
            print(f"Resize error: {e}")

    def release(self):
        self.writer.release()


def process_video(video_path, detection_json, output_dir, video_base_name):
    if not os.path.exists(detection_json):
        return

    with open(detection_json) as f:
        data = json.load(f)

    fps = data["video_info"]["fps"]
    detections = data["detections"]

    cap = cv2.VideoCapture(video_path)
    tracks = []
    next_id = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        boxes = detections.get(str(frame_idx), [])
        used = set()

        for tr in tracks:
            best_iou, best_i = 0, -1
            for i, box in enumerate(boxes):
                if i in used: continue
                score = iou(tr.bbox, box)
                if score > best_iou:
                    best_iou, best_i = score, i

            if best_iou > IOU_THRESHOLD:
                tr.update(boxes[best_i])
                used.add(best_i)
            else:
                tr.miss()

        for i, box in enumerate(boxes):
            if i not in used:
                
                tracks.append(Track(next_id, box, fps, video_base_name, output_dir))
                next_id += 1

        alive = []
        for tr in tracks:
            if not tr.dead():
                tr.write(frame)
                alive.append(tr)
            else:
                tr.release()
        tracks = alive
        frame_idx += 1

    cap.release()
    for tr in tracks:
        tr.release()

    
    candidate_files = []
    for f in os.listdir(output_dir):
        if f.startswith(f"{video_base_name}_temp_") and f.endswith(".avi"):
            candidate_files.append(os.path.join(output_dir, f))

    if len(candidate_files) > 0:
        best_file = max(candidate_files, key=os.path.getsize)
        
        final_path = os.path.join(output_dir, f"{video_base_name}.avi")
        
        if os.path.exists(final_path):
            try: os.remove(final_path)
            except: pass
        
        try:
            os.rename(best_file, final_path)
        except Exception as e:
            print(f"Error renaming {best_file}: {e}")

        for f in candidate_files:
            if f != best_file and os.path.exists(f):
                try: os.remove(f)
                except: pass


def main():
    for cls in CLASSES:
        in_cls = os.path.join(INPUT_ROOT, cls)
        det_cls = os.path.join(DETECTION_ROOT, cls)
        out_cls = os.path.join(OUTPUT_ROOT, cls)
        
        os.makedirs(out_cls, exist_ok=True)

        if not os.path.exists(in_cls): continue

        videos = [v for v in os.listdir(in_cls) if v.endswith(".mp4") and not v.startswith("._")]
        print(f"Processing {len(videos)} videos in '{cls}'...")

        for i, video in enumerate(videos):
            name = os.path.splitext(video)[0]
            
            process_video(
                os.path.join(in_cls, video),
                os.path.join(det_cls, f"{name}.json"),
                out_cls,
                name
            )

    print("\nProcessing Complete.")
    print(f"Check your files in: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
# rPPG-based Face Processing & 3D Model Training

This repository contains the code for a pipeline that detects faces in videos or images, extracts face crops, computes appearance and rPPG features, and trains/tests a 3D classification/regression model.

## Overview

The workflow consists of a sequence of scripts. Each script depends on the output of the previous step:

## NOTE: 
    Make sure that you are inside the main directory where these files are located then only run these commands.It just simplifies the work.

1. **`detect_faces.py`**  
   Entry point for detection.
   - Loads and uses `face_detector.py` to locate faces in the input frames and stores them as JSON.

2. **'extract_crops.py' **
    Using the input as the created JSON files by the detect_faces.py , it extracts the face crops of videos.

3. **`extract_marlin_features.py`**  
   Processes the face crops produced in step 1 and computes MARLIN appearance features.

4. **`extract_superlet.py`**  
   Uses the cropped faces to extract rPPG signals via the Superlet transform.

5. **`script_3d.py`**  
   Consumes the features and rPPG signals to train and evaluate the 3D model.
   The same script can also be used to run inference on a trained network.

## Prerequisites

- Python 3.x
- PyTorch (version compatible with the code)
- OpenCV, NumPy, and any other libraries imported by the scripts
- A CUDA-capable GPU if you intend to train the model

Install dependencies using:

```sh
pip install -r requirements.txt
```

*(Make sure `requirements.txt` lists all necessary packages.)*

## Running the pipeline

1. **Face detection and crop extraction**

   ```sh
   python detect_faces.py

   This will internally load `face_detector.py`. Detected faces  are written to the specified output directory as JSON.

2. **Extracting face crops**
    ,,,
    python extract_crops.py
    ,,,
3. **MARLIN feature extraction**

   ```sh
   python extract_marlin_features.py

4. **rPPG signal extraction**

   ```sh
   python extract_superlet.py
   ```

5. **Train/test the 3D model**

   ```sh
   python script_3d.py
   ```

   Refer to `script_3d.py -h` for available flags and hyperparameters.

## Notes

- Ensure face crop images match the format/size expected by later scripts.
- Keep intermediate directories organized to simplify processing.
- You may need to adjust script parameters or paths for your dataset.

---
*Generated on 26 February 2026*
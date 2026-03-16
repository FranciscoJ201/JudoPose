import os
import sys
from ultralytics import YOLO

# --- Configuration ---
pt_path = "yolov8n-face.pt"  # Place your downloaded face model here
source = 0                   # 0 for webcam, or path to a video file
# ---------------------

if __name__ == "__main__":
    print(f"Loading Model: {pt_path} ...")
    
    # 2. Try CPU (PT File)
    try:
        if os.path.exists(pt_path):
            print(f'Loading CPU model ({pt_path})...')
            
            # Explicitly load the model for CPU
            model = YOLO(pt_path)
            
            print("Model loaded! Starting stream...")
            print("Press 'q' in the video window to quit.")
            
            # Read in the camera or video stream and force CPU
            # stream=True uses a generator for memory efficiency
            results = model.predict(source=source, device="cpu", show=True, stream=True)
            
            # Loop through the stream generator to keep it running
            for r in results:
                pass
                
        else:
            print(f"Error: Could not find model file:\n{pt_path}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Critical: Could not load CPU model: {e}")
        sys.exit(1)
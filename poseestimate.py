import numpy as np
import cv2
from ultralytics import YOLO 
import os
import json 
import torch
import time

def extract_kinematics_from_video(video_path, output_json_path, engine_path='yolo11x-pose.engine'):
    """
    Offline YOLO Pose Extraction.
    Reads a pre-recorded .mp4 video, runs YOLO+BoTSORT tracking, and exports 
    the 2D pixel coordinates in the exact JSON format required by the 3D pipeline.
    """
    
    # --- MODEL LOADING (From Boilerplate) ---
    if not os.path.exists(engine_path):
        if not torch.cuda.is_available():
            print('[YOLO] Swapping to CPU, NO GPU detected')
            model = YOLO('yolo11n-pose.pt')
        else:
            print(f"[YOLO] Exporting engine...")
            model = YOLO('yolo11x-pose.pt')
            model.export(format='engine', half=True)
            model = YOLO(engine_path)
    else:
        print(f"[YOLO] Loading TensorRT engine: {engine_path}")
        model = YOLO(engine_path)

    # --- VIDEO SETUP ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[YOLO] Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[YOLO] Processing {total_frames} frames at {fps} FPS...")

    all_pose_data_for_json = []
    frame_index = 0
    start_time = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # 1. YOLO Inference with BoTSORT tracking
        # We use persist=True to keep track IDs consistent across frames
        results = model.track(source=frame, tracker='botsort.yaml', persist=True, verbose=False)
        frame_detections = [] 

        # 2. Parse Detections if anyone is on screen
        if results and len(results[0].boxes) > 0:
            result = results[0]
            boxes = result.boxes
            
            # Keypoints tensor shape: (num_people, 17, 3)
            keypoints_tensor = result.keypoints.data.cpu().numpy()

            for i in range(len(boxes)):
                # Safely grab track ID (BoTSORT might drop it occasionally)
                track_id = int(boxes.id[i].item()) if boxes.id is not None else -1
                
                person_2d_keypoints = []
                for kp in keypoints_tensor[i]:
                    u, v, conf = float(kp[0]), float(kp[1]), float(kp[2])
                    person_2d_keypoints.append([u, v, conf])

                # Match the exact dictionary structure the pipeline expects
                frame_detections.append({
                    "track_id": track_id,
                    "keypoints_2d": person_2d_keypoints
                })

        # 3. Append to Master List (EVEN IF EMPTY) to preserve timeline sync
        all_pose_data_for_json.append({
            "frame_index": frame_index,
            "orig_hardware_index": frame_index, # Critical for align_jsons.py
            "detections": frame_detections 
        })
        
        # Simple progress logger
        if frame_index % 300 == 0 and frame_index > 0:
            print(f"  Processed {frame_index}/{total_frames} frames...")

        frame_index += 1

    cap.release()
    
    # --- SAVE TO DISK ---
    with open(output_json_path, 'w') as f:
        json.dump(all_pose_data_for_json, f, indent=4)
        
    elapsed = time.perf_counter() - start_time
    print(f"[YOLO] Done! Extracted {frame_index} frames in {elapsed:.1f} seconds.")
    print(f"[YOLO] Saved to {output_json_path}")


if __name__ == "__main__":
    # Example execution:
    # extract_kinematics_from_video(
    #     video_path="realsense_cam_0/skeleton_tracking_output.mp4", 
    #     output_json_path="realsense_cam_0/yolo_output.json"
    # )
    pass
import json
from ultralytics import YOLO
import numpy as np
import os

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def poseestimateGPU(source, engine_path='yolo11x-pose.engine', clinch_iou_threshold=0.3):
    """
    Offline YOLO Pose Extraction with Live Display and Clinch Detection.
    Reads a pre-recorded .mp4, runs YOLO+BoTSORT, shows the live tracking,
    detects clinches based on IoU, and exports the exact JSON format required by the 3D pipeline.
    
    Arguments:
    - source (str): Path to the input video file.
    - engine_path (str): Path to the optimized TensorRT engine.
    - clinch_iou_threshold (float): Intersection over Union threshold to trigger XMem clinch mode.
    """
    # 1. OPTIMIZED MODEL LOADING (From your boilerplate)
    if not os.path.exists(engine_path):
        print(f"Exporting optimized GPU engine: {engine_path}...")
        model = YOLO('yolo11x-pose.pt')
        model.export(format='engine', half=True, device=0) 
        model = YOLO(engine_path)
    else:
        print(f"Loading TensorRT engine: {engine_path}")
        model = YOLO(engine_path)

    base_name = os.path.basename(source)
    video_name, _ = os.path.splitext(base_name)
    
    all_pose_data_for_json = []

    print("GPU Processing started (BoTSORT Tracking + Live Display)...")
    
    # 2. RUN INFERENCE 
    # stream=True processes efficiently, show=True displays the video window.
    results_generator = model.track(
        source=source,
        tracker='botsort.yaml',
        conf=0.3,
        device=0,      # RTX GPU
        half=True,     # FP16 precision
        stream=True,   
        show=True      
    )

    for i, result in enumerate(results_generator):
        frame_detections = []

        # 3. PARSE DETECTIONS (Only if people are found)
        if result.boxes is not None and len(result.boxes) > 0 and result.keypoints is not None:
            keypoints_tensor = result.keypoints.data.cpu().numpy()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy() # Needed for IoU
            boxes_xywh = result.boxes.xywh.cpu().numpy() # Needed for depth_lift.py
            boxes_conf = result.boxes.conf.cpu().numpy() # Needed for depth_lift.py
            
            track_ids = result.boxes.id
            
            # Format track IDs safely (BoTSORT sometimes drops them temporarily)
            if track_ids is None:
                track_ids = [-1] * len(keypoints_tensor)
            else:
                track_ids = track_ids.cpu().numpy().astype(int).tolist()

            # --- CLINCH DETECTION LOGIC ---
            # Check for overlap if there are exactly 2 tracked people (Athlete A and B)
            is_clinching = False
            if len(boxes_xyxy) >= 2:
                iou = calculate_iou(boxes_xyxy[0], boxes_xyxy[1])
                if iou > clinch_iou_threshold:
                    is_clinching = True
                    # TODO: Trigger XMem isolation loop here instead of using raw keypoints.

            for j, keypoint_array in enumerate(keypoints_tensor):
                track_id = track_ids[j] if j < len(track_ids) else -1
                
                # Extract and format strictly to [u, v, conf]
                person_3d_keypoints = [] # Renamed to reflect xyz expectation in depth_lift
                for kp in keypoint_array:
                    u, v, conf = float(kp[0]), float(kp[1]), float(kp[2])
                    person_3d_keypoints.append([u, v, conf])
                    
                frame_detections.append({
                    "track_id_native": track_id,           # Matches depth_lift.py get()
                    "keypoints_xyz": person_3d_keypoints,  # Renamed for depth_lift.py
                    "bbox_xywh": boxes_xywh[j].tolist(),   # Added for depth_lift.py
                    "conf": float(boxes_conf[j]),          # Added for depth_lift.py
                    "is_clinching": is_clinching           # Flag for our upcoming XMem logic
                })

        # 4. CRITICAL: Append EVERY frame to preserve timeline sync
        all_pose_data_for_json.append({
            "frame_index": i,
            "orig_hardware_index": i,
            "detections": frame_detections
        })
        
        if i % 300 == 0 and i > 0:
            print(f"  Processed {i} frames...")

    # 5. SAVE OUTPUT
    output_file = f'{video_name}_pipeline_pose.json'
    with open(output_file, 'w') as f:
        json.dump(all_pose_data_for_json, f, indent=4) 
            
    print(f"\nGPU Finish: Extracted {len(all_pose_data_for_json)} total frames to {output_file}")
    return output_file

if __name__ == "__main__":
    pass
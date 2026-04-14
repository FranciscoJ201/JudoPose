import json
from ultralytics import YOLO
import numpy as np
import os

def poseestimateGPU(source, engine_path='yolo26x-pose.engine'):
    """
    Offline YOLO Pose Extraction with Live Display.
    Reads a pre-recorded .mp4, runs YOLO+BoTSORT, shows the live tracking,
    and exports the exact JSON format required by the 3D pipeline.
    """
    # 1. OPTIMIZED MODEL LOADING (From your boilerplate)
    if not os.path.exists(engine_path):
        print(f"Exporting optimized GPU engine: {engine_path}...")
        model = YOLO('yolo26x-pose.pt')
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
        show=True,
        save = True      
    )

    for i, result in enumerate(results_generator):
        frame_detections = []

        # 3. PARSE DETECTIONS (Only if people are found)
        if result.boxes is not None and len(result.boxes) > 0 and result.keypoints is not None:
            keypoints_tensor = result.keypoints.data.cpu().numpy()
            track_ids = result.boxes.id
            
            # Format track IDs safely (BoTSORT sometimes drops them temporarily)
            if track_ids is None:
                track_ids = [-1] * len(keypoints_tensor)
            else:
                track_ids = track_ids.cpu().numpy().astype(int).tolist()

            for j, keypoint_array in enumerate(keypoints_tensor):
                track_id = track_ids[j] if j < len(track_ids) else -1
                
                # Extract and format strictly to [u, v, conf]
                person_2d_keypoints = []
                for kp in keypoint_array:
                    u, v, conf = float(kp[0]), float(kp[1]), float(kp[2])
                    person_2d_keypoints.append([u, v, conf])
                    
                frame_detections.append({
                    "track_id": track_id,
                    "keypoints_2d": person_2d_keypoints
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
    # Example execution:
    poseestimateGPU(source="/Users/franciscojimenez/Desktop/Screenshot 2026-04-08 at 4.03.01 PM.png")
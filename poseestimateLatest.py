import json
import os
import gc
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  MEMORY MANAGEMENT
# ─────────────────────────────────────────────

def clear_vram(model_variable=None):
    """
    Forces the GPU to completely release a model from VRAM.
    
    Arguments:
    - model_variable: The instantiated model object (e.g., sam_model, yolo_model). 
                      If None, just clears cache.
    """
    if model_variable is not None:
        del model_variable 
    gc.collect() 
    torch.cuda.empty_cache() 

# ─────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────

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

def convert_yolo_xywh_to_sam_xyxy(yolo_xywh):
    """Converts YOLO [Center X, Center Y, W, H] to SAM [x1, y1, x2, y2]."""
    c_x, c_y, w, h = yolo_xywh
    return [c_x - (w / 2.0), c_y - (h / 2.0), c_x + (w / 2.0), c_y + (h / 2.0)]

# ─────────────────────────────────────────────
#  PASS 1: GLOBAL SWEEP (YOLO)
# ─────────────────────────────────────────────

def pass_1_global_sweep(source, engine_path, clinch_iou_threshold):
    """
    Runs YOLO+BoTSORT over the entire video to build the base JSON 
    and identify the exact frames where clinches occur.
    
    Arguments:
    - source (str): Path to input video.
    - engine_path (str): Path to TensorRT engine.
    - clinch_iou_threshold (float): Overlap threshold to trigger a clinch flag.
    
    Returns:
    - list: The full JSON data structure.
    - list: Frame indices where a clinch was detected.
    """
    print("\n--- PASS 1: Global YOLO Sweep ---")
    if not os.path.exists(engine_path):
        print(f"Exporting optimized GPU engine: {engine_path}...")
        model = YOLO('yolo11x-pose.pt')
        model.export(format='engine', half=True, device=0) 
        clear_vram(model)

    print(f"Loading TensorRT engine: {engine_path}")
    model = YOLO(engine_path)
    
    all_pose_data = []
    clinch_frames = []

    results_generator = model.track(
        source=source, tracker='botsort.yaml', conf=0.3, 
        device=0, half=True, stream=True, show=False # Turn off show for speed
    )

    for i, result in enumerate(results_generator):
        frame_detections = []
        is_clinching = False

        if result.boxes is not None and len(result.boxes) > 0 and result.keypoints is not None:
            keypoints_tensor = result.keypoints.data.cpu().numpy()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy() 
            boxes_xywh = result.boxes.xywh.cpu().numpy() 
            boxes_conf = result.boxes.conf.cpu().numpy() 
            track_ids = result.boxes.id
            
            if track_ids is None:
                track_ids = [-1] * len(keypoints_tensor)
            else:
                track_ids = track_ids.cpu().numpy().astype(int).tolist()

            # Clinch Check
            if len(boxes_xyxy) >= 2:
                if calculate_iou(boxes_xyxy[0], boxes_xyxy[1]) > clinch_iou_threshold:
                    is_clinching = True
                    clinch_frames.append(i)

            for j, keypoint_array in enumerate(keypoints_tensor):
                track_id = track_ids[j] if j < len(track_ids) else -1
                
                person_3d_keypoints = []
                for kp in keypoint_array:
                    person_3d_keypoints.append([float(kp[0]), float(kp[1]), float(kp[2])])
                    
                frame_detections.append({
                    "track_id_native": track_id,           
                    "keypoints_xyz": person_3d_keypoints,  
                    "bbox_xywh": boxes_xywh[j].tolist(),   
                    "conf": float(boxes_conf[j]),          
                    "is_clinching": is_clinching           
                })

        # CRITICAL: Append EVERY frame to preserve timeline sync
        all_pose_data.append({
            "frame_index": i,
            "orig_hardware_index": i,
            "detections": frame_detections
        })
        
        if i % 300 == 0 and i > 0: print(f"  Swept {i} frames...")

    clear_vram(model) # Unload YOLO
    return all_pose_data, clinch_frames

# ─────────────────────────────────────────────
#  PASS 2: MASK GENERATION (SAM + XMem)
# ─────────────────────────────────────────────

def pass_2_mask_generation(source, base_json, clinch_frames, sam_weights, mask_out_dir):
    """
    Initializes SAM to generate starting masks, then uses XMem to track 
    them through the clinch sequences.
    
    Arguments:
    - source (str): Video file.
    - base_json (list): The tracking data from Pass 1.
    - clinch_frames (list): The specific frames flagged for clinching.
    - sam_weights (str): Path to SAM checkpoint.
    - mask_out_dir (str): Folder to save XMem blackout masks.
    """
    if not clinch_frames:
        print("\n--- PASS 2: Mask Generation Skipped (No clinches detected) ---")
        return

    print(f"\n--- PASS 2: Mask Generation ({len(clinch_frames)} clinch frames to process) ---")
    os.makedirs(mask_out_dir, exist_ok=True)
    
    # TODO: Initialize SAM Model Here
    # sam = load_sam(sam_weights)
    
    # TODO: Initialize XMem Model Here
    # xmem = load_xmem()
    
    print("  [Placeholder] Running SAM and XMem... saving masks to disk.")
    
    # clear_vram(sam)
    # clear_vram(xmem)

# ─────────────────────────────────────────────
#  PASS 3: BLACKOUT REFINEMENT (YOLO)
# ─────────────────────────────────────────────

def pass_3_blackout_refinement(source, engine_path, base_json, clinch_frames, mask_dir):
    """
    Reloads YOLO to re-evaluate the specific clinch frames using the 
    XMem masks to black out the opponent.
    
    Arguments:
    - source (str): Video file.
    - engine_path (str): TensorRT engine.
    - base_json (list): The tracking data from Pass 1 to be updated.
    - clinch_frames (list): The specific frames flagged for clinching.
    - mask_dir (str): Folder containing XMem blackout masks.
    
    Returns:
    - list: The final, refined JSON data.
    """
    if not clinch_frames:
        return base_json

    print("\n--- PASS 3: Blackout Refinement (YOLO Reloaded) ---")
    model = YOLO(engine_path)
    
    cap = cv2.VideoCapture(source)
    
    for frame_idx in clinch_frames:
        # 1. Read the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        # 2. Load the XMem masks for this frame
        # mask_A = cv2.imread(f"{mask_dir}/mask_A_{frame_idx:06d}.png")
        # mask_B = cv2.imread(f"{mask_dir}/mask_B_{frame_idx:06d}.png")
        
        # 3. Refine Athlete A
        # frame_A = apply_blackout(frame.copy(), mask_B) # Erase B
        # results_A = model.predict(frame_A, verbose=False)
        # Update base_json[frame_idx]['detections'] for Athlete A
        
        # 4. Refine Athlete B
        # frame_B = apply_blackout(frame.copy(), mask_A) # Erase A
        # results_B = model.predict(frame_B, verbose=False)
        # Update base_json[frame_idx]['detections'] for Athlete B
        pass
        
    cap.release()
    clear_vram(model)
    return base_json

# ─────────────────────────────────────────────
#  MASTER ORCHESTRATOR
# ─────────────────────────────────────────────

def poseestimate_multipass(source, engine_path='yolo11x-pose.engine', 
                           sam_checkpoint='sam_vit_h_4b8939.pth', clinch_iou_threshold=0.3):
    """Main execution function that runs the VRAM-optimized multi-pass pipeline."""
    video_name = os.path.splitext(os.path.basename(source))[0]
    output_json = f'{video_name}_pipeline_pose.json'
    mask_dir = f'{video_name}_clinch_masks'
    
    # 1. Global YOLO Sweep
    base_json, clinch_frames = pass_1_global_sweep(source, engine_path, clinch_iou_threshold)
    
    # 2. SAM + XMem Mask Generation
    pass_2_mask_generation(source, base_json, clinch_frames, sam_checkpoint, mask_dir)
    
    # 3. YOLO Blackout Refinement
    final_json = pass_3_blackout_refinement(source, engine_path, base_json, clinch_frames, mask_dir)
    
    # 4. Save Final Output
    with open(output_json, 'w') as f:
        json.dump(final_json, f, indent=4)
        
    print(f"\nPipeline Complete: Extracted {len(final_json)} total frames to {output_json}")
    return output_json


if __name__ == "__main__":
    # poseestimate_multipass(source="realsense_cam_0/skeleton_tracking_output.mp4")
    pass
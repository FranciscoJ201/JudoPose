import json
import os
import gc
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from xmem_bridge import XMemBridge
from segment_anything import sam_model_registry, SamPredictor


def load_sam_predictor(checkpoint_path, model_type="vit_h", device="cuda"):
    """Loads the SAM model into VRAM and returns the Predictor object."""
    print(f"Loading SAM ({model_type}) from {checkpoint_path}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamPredictor(sam)

def generate_sam_masks(predictor, frame_bgr, box_a_xyxy, box_b_xyxy):
    """
    Feeds a frame and two bounding boxes to SAM, returning two precise boolean masks.
    """
    # SAM expects RGB images
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # This calculates the heavy image embeddings (only happens once per frame)
    predictor.set_image(frame_rgb)
    
    # Predict Mask A
    masks_a, _, _ = predictor.predict(
        box=np.array(box_a_xyxy)[None, :],
        multimask_output=False
    )
    
    # Predict Mask B
    masks_b, _, _ = predictor.predict(
        box=np.array(box_b_xyxy)[None, :],
        multimask_output=False
    )
    
    # Return the 2D boolean arrays
    return masks_a[0], masks_b[0]
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
    if not clinch_frames: return

    print(f"\n--- PASS 2: Mask Generation ({len(clinch_frames)} frames) ---")
    os.makedirs(mask_out_dir, exist_ok=True)
    cap = cv2.VideoCapture(source)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    xmem_tracker = None
    current_clinch_event_active = False 

    for frame_idx in clinch_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue

        # --- THE TRIGGER (Initialization) ---
        if not current_clinch_event_active:
            # 1. LOAD SAM ONLY WHEN NEEDED
            sam_predictor = load_sam_predictor(sam_weights, device=device)
            
            boxes = [det['bbox_xywh'] for det in base_json[frame_idx]['detections']]
            box_a_xyxy = convert_yolo_xywh_to_sam_xyxy(boxes[0])
            box_b_xyxy = convert_yolo_xywh_to_sam_xyxy(boxes[1])
            
            mask_a, mask_b = generate_sam_masks(sam_predictor, frame, box_a_xyxy, box_b_xyxy)
            
            # 2. KILL SAM IMMEDIATELY
            clear_vram(sam_predictor.model)
            del sam_predictor
            
            # 3. LOAD XMEM NOW THAT SAM IS GONE
            if xmem_tracker is None:
                xmem_tracker = XMemBridge(checkpoint_path='XMem/XMem.pth', device=device)
            
            xmem_tracker.initialize_masks(frame, mask_a, mask_b)
            current_clinch_event_active = True

        # --- THE TRACKER ---
        out_mask_a, out_mask_b = xmem_tracker.track_frame(frame)
        
        cv2.imwrite(f"{mask_out_dir}/mask_A_{frame_idx:06d}.png", out_mask_a.astype(np.uint8) * 255)
        cv2.imwrite(f"{mask_out_dir}/mask_B_{frame_idx:06d}.png", out_mask_b.astype(np.uint8) * 255)

        if (frame_idx + 1) not in clinch_frames:
            current_clinch_event_active = False
            # Clear memory but keep the model loaded for the next clinch
            xmem_tracker.processor.clear_memory() 

    cap.release()
    if xmem_tracker: clear_vram(xmem_tracker.network)
# ─────────────────────────────────────────────
#  PASS 3: BLACKOUT REFINEMENT (YOLO)
# ─────────────────────────────────────────────

def extract_best_person(yolo_result):
    """
    Helper function to pull the highest-confidence person from a YOLO result.
    Since we blacked out the opponent, there should only be one person left, 
    but this protects against false positives in the background.
    """
    if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
        return None, None, None

    # Get the index of the highest confidence detection
    confidences = yolo_result.boxes.conf.cpu().numpy()
    best_idx = np.argmax(confidences)

    keypoints = yolo_result.keypoints.data[best_idx].cpu().numpy()
    bbox_xywh = yolo_result.boxes.xywh[best_idx].cpu().numpy()
    best_conf = float(confidences[best_idx])

    # Convert keypoints to standard nested list format
    person_3d_keypoints = []
    for kp in keypoints:
        person_3d_keypoints.append([float(kp[0]), float(kp[1]), float(kp[2])])

    return person_3d_keypoints, bbox_xywh.tolist(), best_conf

def pass_3_blackout_refinement(source, engine_path, base_json, clinch_frames, mask_dir, debug_dir="pass3_debug"):
    """
    Reloads YOLO to re-evaluate clinch frames using XMem masks to black out the opponent.
    Includes a visualizer to output side-by-side proofs of the blackout frames.
    """
    if not clinch_frames:
        return base_json

    print(f"\n--- PASS 3: Blackout Refinement ({len(clinch_frames)} frames) ---")
    os.makedirs(debug_dir, exist_ok=True)
    
    model = YOLO(engine_path)
    cap = cv2.VideoCapture(source)
    
    for frame_idx in clinch_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        mask_a_path = f"{mask_dir}/mask_A_{frame_idx:06d}.png"
        mask_b_path = f"{mask_dir}/mask_B_{frame_idx:06d}.png"
        
        if not os.path.exists(mask_a_path) or not os.path.exists(mask_b_path):
            continue
            
        # 1. Load the masks
        mask_a = cv2.imread(mask_a_path, cv2.IMREAD_GRAYSCALE)
        mask_b = cv2.imread(mask_b_path, cv2.IMREAD_GRAYSCALE)
        
        # Make sure masks match the original frame size just in case
        mask_a = cv2.resize(mask_a, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_b = cv2.resize(mask_b, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 2. DIP: The Inversion
        inv_mask_b = cv2.bitwise_not(mask_b)
        inv_mask_a = cv2.bitwise_not(mask_a)
        
        # 3. DIP: The Blackout
        frame_for_a = cv2.bitwise_and(frame, frame, mask=inv_mask_b) # B is erased
        frame_for_b = cv2.bitwise_and(frame, frame, mask=inv_mask_a) # A is erased
        
        # --- NEW: VISUALIZATION EXPORT ---
        # Resize them slightly so the side-by-side image isn't 4000 pixels wide
        vis_h, vis_w = 480, 854 
        vis_frame_a = cv2.resize(frame_for_a.copy(), (vis_w, vis_h))
        vis_frame_b = cv2.resize(frame_for_b.copy(), (vis_w, vis_h))
        
        # Draw labels on them
        cv2.putText(vis_frame_a, "Target: Athlete A (B Blacked Out)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_frame_b, "Target: Athlete B (A Blacked Out)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Glue them together horizontally and save
        debug_image = cv2.hconcat([vis_frame_a, vis_frame_b])
        cv2.imwrite(f"{debug_dir}/blackout_{frame_idx:06d}.jpg", debug_image)
        # ---------------------------------
        
        # 4. Refine Athlete A
        results_a = model.predict(frame_for_a, verbose=False, conf=0.1)[0]
        kp_a, box_a, conf_a = extract_best_person(results_a)
        
        # 5. Refine Athlete B
        results_b = model.predict(frame_for_b, verbose=False, conf=0.1)[0]
        kp_b, box_b, conf_b = extract_best_person(results_b)
        
        # 6. Overwrite the JSON
        if kp_a is not None and len(base_json[frame_idx]['detections']) > 0:
            base_json[frame_idx]['detections'][0]['keypoints_xyz'] = kp_a
            base_json[frame_idx]['detections'][0]['bbox_xywh'] = box_a
            base_json[frame_idx]['detections'][0]['conf'] = conf_a
            
        if kp_b is not None and len(base_json[frame_idx]['detections']) > 1:
            base_json[frame_idx]['detections'][1]['keypoints_xyz'] = kp_b
            base_json[frame_idx]['detections'][1]['bbox_xywh'] = box_b
            base_json[frame_idx]['detections'][1]['conf'] = conf_b

    cap.release()
    print("Flushing YOLO from VRAM...")
    clear_vram(model)
    print(f"Pass 3 Complete. Visual proofs saved to '{debug_dir}'. JSON Refined.")
    
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
    print("--- STARTING PIPELINE TEST ---")
    
    test_video = "C:\\Users\\vrspr\\Downloads\\xmemtest.mp4" # Your short test clip
    
    # Run the multi-pass pipeline
    output_file = poseestimate_multipass(
        source=test_video,
        engine_path="yolo26x-pose.engine",       # Your YOLO weights
        sam_checkpoint="Models/sam_vit_h_4b8939.pth",   # Your SAM weights
        clinch_iou_threshold=0.3                 # XMem trigger threshold
    )
    
    print("--- TEST COMPLETE ---")
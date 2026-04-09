import cv2
import os
import json

def split_video_for_calibration(video_path, output_dir, frame_skip=1):
    """
    Standard blind extraction used ONLY for Stage 2 Calibration (ChArUco boards).
    Extracts every Nth frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_skip == 0:
            out_name = os.path.join(output_dir, f"calib_frame_{saved:06d}.jpg")
            cv2.imwrite(out_name, frame)
            saved += 1
            
        count += 1
        
    cap.release()
    print(f"[VideoSplitter] Calibration split complete. Saved {saved} frames to {output_dir}")


def extract_synced_frames(video_path, synced_json_path, output_dir):
    """
    Extracts perfectly synchronized frames by following the master map 
    created by align_jsons.py. It renames the frames to the unified timeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the synced map
    with open(synced_json_path, 'r') as f:
        synced_data = json.load(f)
        
    # Create a dictionary mapping the original hardware frame to the new unified frame
    # e.g., { 10: 10, 11: 11, 13: 12 } <- Frame 12 was dropped by hardware
    frame_map = {item['orig_hardware_index']: item['frame_index'] for item in synced_data}
    
    cap = cv2.VideoCapture(video_path)
    
    print(f"\n[VideoSplitter] Extracting synced frames using map: {synced_json_path}")
    current_hw_index = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 2. Check if this hardware frame survived the sync cuts
        if current_hw_index in frame_map:
            unified_index = frame_map[current_hw_index]
            
            # 3. Save it under the new unified timeline name
            out_name = os.path.join(output_dir, f"frame_{unified_index:06d}.jpg")
            cv2.imwrite(out_name, frame)
            saved_count += 1
            
        current_hw_index += 1
        
    cap.release()
    print(f"[VideoSplitter] Success. Extracted {saved_count} perfectly synced frames to {output_dir}")

if __name__ == "__main__":
    # Example usage:
    # After you run align_jsons.py, you pass the raw video and the cleaned JSON here.
    
    # extract_synced_frames(
    #     video_path="realsense_cam_0/color.mp4",
    #     synced_json_path="synced_yolo_output/realsense_0_time_synced.json",
    #     output_dir="synced_frames/realsense_0"
    # )
    pass
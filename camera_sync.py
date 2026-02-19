import cv2
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor

def find_flash_frame(video_path, scan_duration_sec=100):
    """
    Finds a flash by looking for the largest sudden *jump* in average brightness,
    ignoring static white objects like walls or Judo Gis.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * scan_duration_sec)
    
    brightness_values = []
    
    print(f"Scanning {os.path.basename(video_path)} for flash...")
    
    # 1. Record the average brightness of every frame
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use mean() instead of max(). A flash lights up the whole room/lens.
        brightness_values.append(np.mean(gray)) 
            
    cap.release()
    
    if len(brightness_values) < 2:
        return 0

    # 2. Calculate the difference between consecutive frames
    # np.diff subtracts frame 1 from frame 2, frame 2 from frame 3, etc.
    brightness_jumps = np.diff(brightness_values)
    
    # 3. The flash is the frame with the biggest positive jump
    flash_frame_idx = np.argmax(brightness_jumps) + 1 
    biggest_jump = brightness_jumps[flash_frame_idx - 1]
    
    # 4. Sanity Check: Was it an actual flash or just normal movement?
    # We require the jump to be significant (e.g., average pixel brightness jumped by at least 5)
    if biggest_jump < 5.0: 
        print(f"  Warning: No clear flash jump detected in {os.path.basename(video_path)} (Max jump: {biggest_jump:.2f})")
        return 0
    
    print(f"  -> Flash found at Frame {flash_frame_idx} (Brightness jump: +{biggest_jump:.2f})")
    return flash_frame_idx

def process_single_video(video_path, output_folder):
    """
    Worker function to sync and save one video.
    """
    filename = os.path.basename(video_path)
    out_path = os.path.join(output_folder, f"synced_{filename}")
    
    # 1. Find the start point
    start_frame = find_flash_frame(video_path)
    
    # 2. Re-save the video starting from that frame
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    # Skip to the flash
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"Writing {filename} starting from frame {start_frame}...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
        
        if frame_count % 500 == 0:
            print(f"  {filename}: {frame_count}/{(total_frames-start_frame)} frames")

    cap.release()
    out.release()
    print(f"Finished {filename}")

def sync_batch(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    
    # Process videos in parallel (fast!)
    with ThreadPoolExecutor() as executor:
        for video in video_files:
            executor.submit(process_single_video, video, output_folder)

if __name__ == "__main__":
    # CHANGE THESE PATHS
    INPUT_DIR = "/Users/franciscojimenez/Desktop"
    OUTPUT_DIR = "synced_videos"
    
    sync_batch(INPUT_DIR, OUTPUT_DIR)
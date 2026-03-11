import pyrealsense2 as rs
import numpy as np
import cv2
import time

# --- CONFIGURATION ---
OUTPUT_FILENAME = 'realsense_color_only.mp4'
W, H = 848, 480 
TARGET_FPS = 59.9400000

# --- VIDEO WRITER CONFIG ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, TARGET_FPS, (W, H))

# --- REALSENSE SETUP ---
pipeline = rs.pipeline()
config = rs.config()

# We only enable the color stream here, no depth needed
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, TARGET_FPS)

print(f"Starting Realsense stream at {W}x{H}, {TARGET_FPS} FPS...")
pipeline.start(config)

try:
    frame_count = 0
    start_time = time.time()
    
    print("Recording... Press 'q' to stop.")
    
    while True:
        # Wait for the next set of frames from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        # Convert the Realsense frame to a numpy array for OpenCV
        color_image = np.asanyarray(color_frame.get_data())
        
        # Save the frame to our video file
        out.write(color_image)
        
        # Optional: Add a simple frame counter to the visual display
        cv2.putText(color_image, f"Recording: {frame_count} frames", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the video stream in a window
        cv2.imshow("RealSense Color Stream", color_image)
        
        frame_count += 1

        # Check if the 'q' key was pressed to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Calculate and print the actual recorded FPS at the end
    elapsed_time = time.time() - start_time
    actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    # Clean everything up
    pipeline.stop()
    out.release() 
    cv2.destroyAllWindows()
    
    print(f"Finished! Saved {frame_count} frames to '{OUTPUT_FILENAME}'.")
    print(f"Actual recorded speed: {actual_fps:.2f} FPS")
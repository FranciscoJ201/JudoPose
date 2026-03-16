import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading
import queue
import sys

# --- CONFIGURATION ---
OUTPUT_FILENAME = 'realsense_color_only.mp4'
W, H = 848, 480 
TARGET_FPS = 60

# --- VIDEO WRITER CONFIG ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, TARGET_FPS, (W, H))

def video_writer_thread(frame_queue, writer, stop_flag):
    """
    Consumer thread that fetches frames from the queue and encodes them to the video file.
    
    Arguments:
    - frame_queue (queue.Queue): Thread-safe queue containing numpy image arrays.
    - writer (cv2.VideoWriter): The OpenCV object used to save the video.
    - stop_flag (threading.Event): Event used to signal the thread that recording is over.
    """
    while not stop_flag.is_set() or not frame_queue.empty():
        try:
            # Block for up to 0.1s waiting for a frame
            frame = frame_queue.get(timeout=0.1)
            writer.write(frame)
            frame_queue.task_done()
        except queue.Empty:
            continue

# --- REALSENSE SETUP ---
pipeline = rs.pipeline()
config = rs.config()

# We only enable the color stream here, no depth needed
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, TARGET_FPS)

print(f"Starting Realsense stream at {W}x{H}, {TARGET_FPS} FPS...")
pipeline.start(config)

# Maxsize 300 buffers up to 5 seconds of frames if the writer gets behind
frame_buffer = queue.Queue(maxsize=300)
recording_stopped = threading.Event()

# Start the background writing thread
writer_thread = threading.Thread(target=video_writer_thread, args=(frame_buffer, out, recording_stopped))
writer_thread.start()

try:
    print("Showing preview. Press 'c' to start recording headlessly, or 'q' to quit.")
    
    # --- PREVIEW LOOP ---
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
            
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("Preview - Press 'c' to Record", color_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("Closing preview and starting headless recording...")
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
            print("Exiting before recording...")
            recording_stopped.set()
            writer_thread.join()
            pipeline.stop()
            out.release()
            sys.exit(0)
            
    # --- MAIN RECORDING LOOP ---
    frame_count = 0
    start_time = time.time()
    
    print("Recording headlessly... Press Ctrl+C in the terminal to stop.")
    
    while True:
        # Wait for the next set of frames from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        # Convert the Realsense frame to a numpy array for OpenCV
        color_image = np.asanyarray(color_frame.get_data())
        
        # Save the frame to our video file (Passed to the queue thread)
        try:
            frame_buffer.put(color_image, block=False)
            frame_count += 1
        except queue.Full:
            print("Warning: Queue is full! Dropping frame.")
        
        # Optional: Add a simple frame counter to the visual display
        # (Kept original comment, but text logic removed since display is closed)
        
        # Show the video stream in a window
        # (Kept original comment, but imshow removed for headless performance)
        
        # Check if the 'q' key was pressed to break the loop
        # (Kept original comment, but waitKey disabled. Relying on KeyboardInterrupt instead.)

except KeyboardInterrupt:
    print("\nCtrl+C detected! Stopping recording cleanly...")

finally:
    # Signal the writing thread to finish emptying the queue and wait for it
    recording_stopped.set()
    writer_thread.join()

    # Calculate and print the actual recorded FPS at the end
    elapsed_time = time.time() - start_time
    actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    # Clean everything up
    pipeline.stop()
    out.release() 
    
    # cv2.destroyAllWindows() is already handled, but safe to call again
    cv2.destroyAllWindows()
    
    print(f"Finished! Saved {frame_count} frames to '{OUTPUT_FILENAME}'.")
    print(f"Actual recorded speed: {actual_fps:.2f} FPS")
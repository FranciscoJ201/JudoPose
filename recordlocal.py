import cv2
import time

def record_basic_camera(camera_index=0, output_filename="output.mp4"):
    """
    Records video from a standard USB camera input using its native resolution and FPS.
    """
    # 1. Initialize the camera
    print(f"Connecting to camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}.")
        print("Tip: If 0 doesn't work, try 1, 2, or 3.")
        return

    # 2. Get Camera Hardware Settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # OpenCV USB Quirk: Sometimes USB cameras report 0 FPS. 
    # If so, we force a standard fallback so the video writer doesn't crash.
    if fps == 0 or fps == -1:
        print("Warning: Camera did not report FPS. Defaulting to 30.0 FPS.")
        fps = 30.0
    else:
        print(f"Camera Hardware Detected: {width}x{height} @ {fps} FPS")

    # 3. Setup the Video Writer
    # 'mp4v' is the standard codec for .mp4 files in OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"\n--- RECORDING TO {output_filename} ---")
    print(">>> FLICK YOUR FLASHLIGHT NOW FOR SYNC <<<")
    print("Press 'q' on your keyboard to stop recording.\n")

    frames_recorded = 0
    start_time = time.time()

    # 4. The Recording Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera disconnected or frame dropped.")
            break

        # Write the frame to the file
        out.write(frame)
        frames_recorded += 1

        # Show the live feed on your screen
        cv2.imshow(f"Camera {camera_index} Recording", frame)

        # Listen for the 'q' key to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 5. Cleanup
    end_time = time.time()
    duration = end_time - start_time
    actual_fps = frames_recorded / duration

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n--- RECORDING FINISHED ---")
    print(f"Total Frames: {frames_recorded}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Actual Captured FPS: {actual_fps:.2f}")
    
    # Sanity check for dropped frames
    if abs(actual_fps - fps) > 5:
        print(f"WARNING: Your target FPS was {fps}, but you captured at {actual_fps:.2f}.")
        print("Your CPU/Hard Drive might be too slow to save the frames in real-time.")

if __name__ == "__main__":
    # Change the index (0, 1, 2) depending on which USB port the camera is in.
    record_basic_camera(camera_index=0, output_filename="gopro_raw.mp4")
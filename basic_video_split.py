import cv2
import os

def split_video_for_calibration(video_path, output_dir, frame_skip=30):
    """
    Extracts frames from a video for calibration, skipping frames to avoid 
    clogging the math solver with thousands of nearly identical images.
    
    Args:
        video_path (str): Path to your .mp4 file.
        output_dir (str): Folder to save the .jpg images.
        frame_skip (int): Saves 1 out of every X frames. (e.g., 30 means 1 frame per second for a 30fps video).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get total frames to show a progress estimate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Opened {video_path} ({total_frames} frames @ {fps} FPS)")
    
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only save every Nth frame
        if frame_count % frame_skip == 0:
            filename = os.path.join(output_dir, f"calib_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done! Extracted {saved_count} distinct frames into '{output_dir}'.")


if __name__ == "__main__":
    # Example usage:
    # If shooting at 60fps, setting frame_skip=30 saves 2 images per second.
    
    # split_video_for_calibration("gopro1_checkerboard.mp4", "frames_gopro1", frame_skip=30)
    # split_video_for_calibration("realsense_checkerboard.mp4", "frames_realsense", frame_skip=15)
    split_video_for_calibration('gopro_raw.mp4','image_folder',frame_skip=10)
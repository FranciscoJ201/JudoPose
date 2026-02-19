import cv2

def record_video(output_path):
    # 0 is usually the default built-in webcam. 
    # Change to 1 or 2 if you have an external USB camera.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Grab the default width and height of the camera's frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0  # Frames per second

    # Define the codec and create a VideoWriter object. 
    # 'mp4v' is great for saving as an .mp4 file.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Recording... Saving to '{output_path}'.")
    print("Press 'q' on your keyboard to stop and quit.")

    while True:
        # Read the camera frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read the frame. Exiting...")
            break

        # Save the frame to our video file
        out.write(frame)

        # Show the video stream in a window so you can see what's recording
        cv2.imshow('Webcam Recording', frame)

        # Wait for 1 millisecond and check if the 'q' key was pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Once 'q' is pressed, clean everything up and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video successfully saved!")

if __name__ == "__main__":
    # Specify your desired file path and name here
    save_path = "my_recording.mp4" 
    record_video(save_path)
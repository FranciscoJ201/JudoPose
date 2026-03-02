from video_split import split_video_for_calibration
from intrinsic import calibrate_camera_charuco
from extrinsic import calibrate_extrinsics_pnp
import os

if __name__ == "__main__":
    # split_video_for_calibration('240.mp4','image_folder4',frame_skip=1)
    # split_video_for_calibration('120.mp4','image_folder5',frame_skip=1)
    split_video_for_calibration('please20.mp4','image_folder10',frame_skip=1)

    
    # RealSense standard calibration
    # calibrate_camera_charuco(
    #     image_folder="image_folder4", 
    #     output_file="intrinsic_realsense4.json", 
    #     is_fisheye=True # Triggers the Rational Model for high distortion
    # )
    # calibrate_camera_charuco(
    #     image_folder="image_folder5", 
    #     output_file="intrinsic_realsense5.json", 
    #     is_fisheye=True # Triggers the Rational Model for high distortion
    # )
    calibrate_camera_charuco(
        image_folder="image_folder10", 
        output_file="please.json", 
        is_fisheye=True # Triggers the Rational Model for high distortion
    )
    calibrate_extrinsics_pnp("/Users/franciscojimenez/Desktop/JudoPose/testingground/image_folder10/calib_0500.jpg", "please.json", "extrinsic_gopro1.json", mat_size_meters=8.0)
    
 
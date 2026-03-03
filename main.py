from video_split import split_video_for_calibration
from intrinsic import calibrate_camera_charuco
from extrinsic import calibrate_extrinsics_pnp
import os

if __name__ == "__main__":
    vid1 = 'test1.mp4'
    vid2 = 'lineartest.mp4'
    folder1 = 'image_folder1'
    folder2 = 'image_folder2'
    intrOUT1 = "intrinsic1.json"
    intrOUT2 = "intrinsic2.json"

    split_video_for_calibration(vid1,folder1,frame_skip=10)
    split_video_for_calibration(vid2,folder2,frame_skip=10)

    
    # RealSense standard calibration
    calibrate_camera_charuco(
        image_folder="image_folder1", 
        output_file=intrOUT1, 
        is_fisheye=True, # Triggers the Rational Model for high distortion
        show_graph=True
    )
    calibrate_camera_charuco(
        image_folder="image_folder2", 
        output_file=intrOUT2, 
        is_fisheye=False, # Triggers the Rational Model for high distortion
        show_graph=True
    )
    # calibrate_camera_charuco(
    #     image_folder="image_folder", 
    #     output_file="please.json", 
    #     is_fisheye=False, # Triggers the Rational Model for high distortion,
    #     show_graph= True
    # )
    calibrate_extrinsics_pnp(f"{folder1}/calib_0085.jpg", intrOUT1, "extrinsic1.json", mat_size_meters=8.0)
    calibrate_extrinsics_pnp(f"{folder2}/calib_0085.jpg", intrOUT2, "extrinsic2.json", mat_size_meters=8.0)

    
 
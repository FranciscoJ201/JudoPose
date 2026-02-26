from intrinsic import calibrate_camera_charuco

from intrinsic import calibrate_camera_charuco
from extrinsic import calibrate_extrinsics_pnp
import os

def run_calibration_pipeline():
    print("=== JUDO 3D: CALIBRATION CONTROL CENTER ===")

    # ---------------------------------------------------------
    # STEP 1: INTRINSIC CALIBRATION (The Lens Math)
    # Do this once per camera. You can reuse these JSONs forever 
    # unless you change the lens or GoPro FOV setting.
    # ---------------------------------------------------------
    
    # RealSense (Standard Lens)
    calibrate_camera_charuco(
        image_folder="image_folder", 
        output_file="intrinsic_realsense.json", 
        is_fisheye=False
    )

    # GoPro (Fisheye Lens)
    # calibrate_camera_charuco(
    #     image_folder="frames_gopro1", 
    #     output_file="intrinsic_gopro1.json", 
    #     is_fisheye=True
    # )


    # ---------------------------------------------------------
    # STEP 2: EXTRINSIC CALIBRATION (The Room Setup)
    # Do this every time you set up the tripods around the mat.
    # ---------------------------------------------------------
    
    # RealSense Position
    calibrate_extrinsics_pnp(
        image_path="image_folder/calib_0000.jpg", 
        intrinsic_json="intrinsic_realsense.json", 
        output_json="extrinsic_realsense.json", 
        mat_size_meters=0.0254 # Standard combat area
    )

    # GoPro Position
    # calibrate_extrinsics_pnp(
    #     image_path="gopro1_empty_mat.jpg", 
    #     intrinsic_json="intrinsic_gopro1.json", 
    #     output_json="extrinsic_gopro1.json", 
    #     mat_size_meters=8.0 
    # )

    print("\nCalibration Pipeline Complete.")
    print("If all JSONs are generated, you are ready for Triangulation.")

if __name__ == "__main__":
    run_calibration_pipeline()
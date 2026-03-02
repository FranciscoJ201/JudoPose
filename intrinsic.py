import cv2
import numpy as np
import glob
import json
import os

def calibrate_camera_charuco(image_folder, output_file, is_fisheye=False):
    """
    Calibrates a camera using a ChArUco board. 
    Routes data to standard math (RealSense) or fisheye math (GoPro Rational Model).
    """
    print(f"\n--- Starting Calibration ---")
    print(f"Directory: {image_folder}")
    print(f"Model: {'FISHEYE (GoPro Rational)' if is_fisheye else 'STANDARD (RealSense)'}")

    # --- 1. Define the ChArUco Board ---
    # MAKE SURE THESE MATCH YOUR PRINTED BOARD EXACTLY
    SQUARES_X = 11
    SQUARES_Y = 8
    SQUARE_LENGTH = 0.01  # 30mm in meters
    MARKER_LENGTH = 0.0075 # 23mm in meters
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    
    # NEW API: Setup the dedicated ChArUco detector
    charuco_detector = cv2.aruco.CharucoDetector(board)

    # --- 2. Gather Data ---
    all_corners = []
    all_ids = []
    image_size = None
    
    images = glob.glob(os.path.join(image_folder, '*.jpg'))
    if not images:
        print(f"Error: No images found in {image_folder}")
        return

    print(f"Processing {len(images)} images...")
    
    # NEW: Create a directory to save the valid frames for visual inspection
    valid_folder = f"{image_folder}_valid"
    os.makedirs(valid_folder, exist_ok=True)
    print(f"Valid frames will be saved to: {valid_folder}")
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray.shape[::-1] # (width, height)
            
        # detectBoard handles the markers and the interpolation all at once
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        
        # SAFETY CHECK: Require at least 12 corners to satisfy the 18-variable Rational Model
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) >= 12:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            
            # NEW: Save the successful frame
            # cv2.imwrite arguments:
            # - filename (str): The path/name of the file to save
            # - img (numpy.ndarray): The image array to save
            base_name = os.path.basename(fname)
            save_path = os.path.join(valid_folder, base_name)
            cv2.imwrite(save_path, img)
            
        else:
            # Tell the user we are skipping a bad frame so they know the script isn't broken
            corners_found = len(charuco_corners) if charuco_corners is not None else 0
            # print(f"  Skipping {os.path.basename(fname)}: Only {corners_found} corners found (Need >= 12)")

    if len(all_corners) == 0:
        print("Error: Could not find enough ChArUco corners in the images.")
        return

    print(f"\nUsable frames extracted: {len(all_corners)} out of {len(images)}")

    # --- 3. Format Data for the Solvers ---
    objpoints = []
    imgpoints = []
    
    board_3d_points = board.getChessboardCorners()
    
    for i in range(len(all_corners)):
        corners_2d = all_corners[i]
        ids = all_ids[i].flatten()
        corners_3d = board_3d_points[ids]
        
        imgpoints.append(corners_2d)
        objpoints.append(corners_3d)

    # --- 4. The Math (Standard vs Fisheye) ---
    print("Running mathematical solvers...")
    
    if is_fisheye:
        # Changed to Rational Model for GoPro to avoid cv2.fisheye crashes
        flags = cv2.CALIB_RATIONAL_MODEL
        
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=flags
        )
        camera_model = "fisheye_rational"
        
    else:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None
        )
        camera_model = "standard"

    # --- 5. Save the Results ---
    calibration_data = {
        "camera_model": camera_model,
        "reprojection_error": float(rms),
        "camera_matrix": K.tolist(),
        "distortion_coefficients": D.tolist(),
        "image_size": image_size
    }

    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=4)

    print(f"SUCCESS! Reprojection Error: {rms:.4f} pixels")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    pass
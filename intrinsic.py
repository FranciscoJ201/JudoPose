import cv2
import numpy as np
import glob
import json
import os

def calibrate_camera_charuco(image_folder, output_file, is_fisheye=False):
    """
    Calibrates a camera using a ChArUco board. 
    Routes data to standard math (RealSense) or fisheye math (GoPro).
    """
    print(f"\n--- Starting Calibration ---")
    print(f"Directory: {image_folder}")
    print(f"Model: {'FISHEYE (GoPro)' if is_fisheye else 'STANDARD (RealSense)'}")

    # --- 1. Define the ChArUco Board ---
    # MAKE SURE THESE MATCH YOUR PRINTED BOARD EXACTLY
    SQUARES_X = 11
    SQUARES_Y = 8
    SQUARE_LENGTH = 0.015  # 30mm in meters
    MARKER_LENGTH = 0.011  # 23mm in meters
    
    # We use a standard dictionary (4x4 markers)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    
    # Setup the ArUco detector
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    # --- 2. Gather Data ---
    all_corners = []
    all_ids = []
    image_size = None
    
    images = glob.glob(os.path.join(image_folder, '*.jpg'))
    if not images:
        print(f"Error: No images found in {image_folder}")
        return

    print(f"Processing {len(images)} images...")
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray.shape[::-1] # (width, height)

        # Detect raw ArUco markers
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        
        if len(marker_corners) > 0:
            # Interpolate the inner checkerboard corners based on the markers
            # This works even if half the board is hidden or off-screen
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )
            
            # Need at least 4 corners to do any meaningful math
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

    if len(all_corners) == 0:
        print("Error: Could not find enough ChArUco corners in the images.")
        return

    # --- 3. Format Data for the Solvers ---
    # We must extract the exact 3D coordinates for the specific corners detected in each frame
    objpoints = []
    imgpoints = []
    
    board_3d_points = board.getChessboardCorners()
    
    for i in range(len(all_corners)):
        # The 2D pixels found in the image
        corners_2d = all_corners[i]
        # The IDs of those corners
        ids = all_ids[i].flatten()
        # The corresponding 3D physical locations of those specific corners on the board
        corners_3d = board_3d_points[ids]
        
        if is_fisheye:
            # FISHEYE GOTCHA: Must reshape arrays to strictly (N, 1, 2) and (N, 1, 3)
            imgpoints.append(corners_2d.reshape(-1, 1, 2))
            objpoints.append(corners_3d.reshape(-1, 1, 3))
        else:
            imgpoints.append(corners_2d)
            objpoints.append(corners_3d)

    # --- 4. The Math (Standard vs Fisheye) ---
    print("Running mathematical solvers...")
    
    if is_fisheye:
        # Fisheye flags to prevent infinite distortion math loops
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
        
        # Initialize empty matrices
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        
        # Calculate!
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, image_size, K, D, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        camera_model = "fisheye"
        
    else:
        # Standard Brown-Conrady model
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
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
    # Example usage:
    # 1. Calibrate the RealSense (Standard)
    # calibrate_camera_charuco("frames_realsense", "intrinsic_realsense.json", is_fisheye=False)
    
    # 2. Calibrate a GoPro (Fisheye)
    # calibrate_camera_charuco("frames_gopro1", "intrinsic_gopro1.json", is_fisheye=True)
    pass
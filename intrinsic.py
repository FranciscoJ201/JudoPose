import cv2
import numpy as np
import glob
import json
import os
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm

def plot_calibration_graphs(objpoints, rvecs, tvecs, imgpoints, image_size):
    """
    Plots both the 3D physical positions of the ChArUco board and a 2D heatmap
    of the pixel coverage on the camera sensor in a single window.
    
    Arguments:
    - objpoints: List of 3D physical board coordinates.
    - rvecs: Rotation vectors from OpenCV.
    - tvecs: Translation vectors from OpenCV.
    - imgpoints: Raw 2D pixel coordinates of the board on the sensor.
    - image_size: (width, height) tuple of the camera resolution.
    """
    # Create a single large window with two subplots (1 row, 2 columns)
    fig = plt.figure(figsize=(16, 8))
    
    # --- Subplot 1: The 3D Board Positions (Left) ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot the camera lens as a black triangle at the origin
    ax1.scatter(0, 0, 0, c='black', marker='^', s=200, label='Camera Lens')
    
    # Loop through every valid frame that passed the calibration math
    for i in range(len(objpoints)):
        R, _ = cv2.Rodrigues(rvecs[i])
        t = tvecs[i]
        
        # Reshape and transform the local board points into the camera's 3D space
        pts = objpoints[i].reshape(-1, 3)
        transformed_pts = (R @ pts.T) + t
        transformed_pts = transformed_pts.T
        
        label_name = f'Frame {i+1}' if i < 15 else ""
        ax1.scatter(transformed_pts[:, 0], transformed_pts[:, 1], transformed_pts[:, 2], 
                   alpha=0.6, s=15, label=label_name)
        
    ax1.set_xlabel('X Axis (Meters)')
    ax1.set_ylabel('Y Axis (Meters)')
    ax1.set_zlabel('Z Axis (Depth from Lens in Meters)')
    ax1.set_title('3D ChArUco Board Positions')
    
    handles, labels = ax1.get_legend_handles_labels()
    if len(handles) > 15:
        ax1.legend(handles[:15], labels[:15])
    else:
        ax1.legend()

    # --- Subplot 2: The 2D Pixel Density Heatmap (Right) ---
    ax2 = fig.add_subplot(122)
    
    # Flatten all 2D image points into a single X list and Y list
    all_x = []
    all_y = []
    for corners in imgpoints:
        # OpenCV corner shape is usually (N, 1, 2)
        for corner in corners:
            all_x.append(corner[0][0])
            all_y.append(corner[0][1])
            
    width, height = image_size
    
    # Create the color-coded density map (heatmap)
    # Bins divide the screen into a grid to count corners in each sector
    h = ax2.hist2d(all_x, all_y, bins=[int(width/20), int(height/20)], 
                   range=[[0, width], [0, height]], cmap= 'inferno',norm = LogNorm())
    
    # Add a color bar legend to the side
    fig.colorbar(h[3], ax=ax2, label='Corner Density (Hotter = More Data)')
    
    # Invert the Y axis so 0 is at the top (exactly like a real image)
    ax2.invert_yaxis()
    
    ax2.set_xlabel('Image Width (Pixels)')
    ax2.set_ylabel('Image Height (Pixels)')
    ax2.set_title('2D Image Plane Heatmap ')
    ax2.set_xlim(0, width)
    ax2.set_ylim(height, 0)
    
    plt.tight_layout()
    plt.show()


def calibrate_camera_charuco(image_folder, output_file, is_fisheye=False, show_graph=False):
    """
    Calibrates a camera using a ChArUco board. 
    Routes data to standard math (RealSense) or fisheye math (GoPro Rational Model).
    
    Arguments:
    - image_folder: String path to the folder containing extracted frames.
    - output_file: String path for saving the resulting JSON matrix.
    - is_fisheye: Boolean to toggle the Rational Model for ultra-wide lenses.
    - show_graph: Boolean to display the 3D position and 2D heatmap window.
    """
    print(f"\n--- Starting Calibration ---")
    print(f"Directory: {image_folder}")
    print(f"Model: {'FISHEYE (GoPro Rational)' if is_fisheye else 'STANDARD (RealSense)'}")

    # --- 1. Define the ChArUco Board ---
    # MAKE SURE THESE MATCH YOUR PRINTED BOARD EXACTLY
    SQUARES_X = 5
    SQUARES_Y = 7
    SQUARE_LENGTH = 0.04  # 40mm in meters
    MARKER_LENGTH = 0.03  # 30mm in meters
    
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
            
            base_name = os.path.basename(fname)
            save_path = os.path.join(valid_folder, base_name)
            cv2.imwrite(save_path, img)

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

    print(f"SUCCESS! Reprojection Error: {rms:.4f} pixels")

    # --- 5. Optional 3D Visualization ---
    if show_graph:
        print("Generating 3D Visualization...")
        # Now passing imgpoints and image_size to drive the new heatmap
        plot_calibration_graphs(objpoints, rvecs, tvecs, imgpoints, image_size)

    # --- 6. Save the Results ---
    calibration_data = {
        "camera_model": camera_model,
        "reprojection_error": float(rms),
        "camera_matrix": K.tolist(),
        "distortion_coefficients": D.tolist(),
        "image_size": image_size
    }

    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=4)

    print(f"Saved to {output_file}")

if __name__ == "__main__":
    pass
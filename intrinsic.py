import cv2
import numpy as np
import glob
import json
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # NEW: Boosts visibility of low-data spots

def plot_calibration_graphs(objpoints, rvecs, tvecs, raw_imgpoints, imgpoints, image_size):
    """
    Plots a 3-panel dashboard: 3D board positions, the raw (unfiltered) 2D pixel density, 
    and the final (filtered) 2D pixel density.
    
    Arguments:
    - objpoints: List of 3D physical board coordinates.
    - rvecs: Rotation vectors from OpenCV.
    - tvecs: Translation vectors from OpenCV.
    - raw_imgpoints: The massive, unfiltered list of 2D corners (Before Filter).
    - imgpoints: The strictly filtered list of 2D corners (After Filter).
    - image_size: (width, height) tuple of the camera resolution.
    """
    # Create a massive window with three subplots (1 row, 3 columns)
    fig = plt.figure(figsize=(24, 8))
    
    # --- Panel 1: The 3D Board Positions (Left) ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(0, 0, 0, c='black', marker='^', s=200, label='Camera Lens')
    
    for i in range(len(objpoints)):
        R, _ = cv2.Rodrigues(rvecs[i])
        t = tvecs[i]
        pts = objpoints[i].reshape(-1, 3)
        transformed_pts = (R @ pts.T) + t
        transformed_pts = transformed_pts.T
        
        label_name = f'Frame {i+1}' if i < 15 else ""
        ax1.scatter(transformed_pts[:, 0], transformed_pts[:, 1], transformed_pts[:, 2], 
                   alpha=0.6, s=15, label=label_name)
        
    ax1.set_xlabel('X Axis (Meters)')
    ax1.set_ylabel('Y Axis (Meters)')
    ax1.set_zlabel('Z Axis (Depth from Lens in Meters)')
    ax1.set_title('Final 3D ChArUco Positions')
    
    handles, labels = ax1.get_legend_handles_labels()
    if len(handles) > 15:
        ax1.legend(handles[:15], labels[:15])
    else:
        ax1.legend()

    # --- Helper Function for Heatmaps ---
    def draw_heatmap(ax, points, title):
        all_x, all_y = [], []
        for corners in points:
            for corner in corners:
                all_x.append(corner[0][0])
                all_y.append(corner[0][1])
                
        width, height = image_size
        
        # Using LogNorm() so even 1 single frame glows brightly against the black
        h = ax.hist2d(all_x, all_y, bins=[int(width/20), int(height/20)], 
                      range=[[0, width], [0, height]], cmap='inferno', norm=LogNorm())
        
        fig.colorbar(h[3], ax=ax, label='Logarithmic Corner Density')
        ax.invert_yaxis()
        ax.set_xlabel('Image Width (Pixels)')
        ax.set_ylabel('Image Height (Pixels)')
        ax.set_title(title)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

    # --- Panel 2: Before Filter Heatmap (Center) ---
    ax2 = fig.add_subplot(132)
    draw_heatmap(ax2, raw_imgpoints, f'BEFORE Filter ({len(raw_imgpoints)} Frames)')

    # --- Panel 3: After Filter Heatmap (Right) ---
    ax3 = fig.add_subplot(133)
    draw_heatmap(ax3, imgpoints, f'AFTER 10x10 Grid Filter ({len(imgpoints)} Frames)')
    
    plt.tight_layout()
    plt.show()


def calibrate_camera_charuco(image_folder, output_file, is_fisheye=False, show_graph=False):
    """
    Calibrates a camera using a ChArUco board. 
    Routes data to standard math (RealSense) or fisheye math (GoPro Rational Model).
    """
    print(f"\n--- Starting Calibration ---")
    print(f"Directory: {image_folder}")
    print(f"Model: {'FISHEYE (GoPro Rational)' if is_fisheye else 'STANDARD (RealSense)'}")

    # --- 1. Define the ChArUco Board ---
    SQUARES_X = 5
    SQUARES_Y = 7
    SQUARE_LENGTH = 0.04  
    MARKER_LENGTH = 0.03 
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    charuco_detector = cv2.aruco.CharucoDetector(board)

    # --- 2. Gather Data & Apply 10x10 Spatial Filter ---
    all_corners = []
    all_ids = []
    raw_imgpoints = [] # Holds every frame seen for the "Before" heatmap
    image_size = None
    
    # Filter Constants
    GRID_COLS = 10
    GRID_ROWS = 10
    MAX_PER_BIN = 7
    grid_counts = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    
    images = glob.glob(os.path.join(image_folder, '*.jpg'))
    if not images:
        print(f"Error: No images found in {image_folder}")
        return

    print(f"Processing {len(images)} images through 10x10 filter...")
    
    valid_folder = f"{image_folder}_valid"
    os.makedirs(valid_folder, exist_ok=True)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray.shape[::-1]
            
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) >= 12:
            # Save to raw list immediately for the "Before" graph
            raw_imgpoints.append(charuco_corners)
            
            # --- THE GATEKEEPER ---
            # 1. Calculate the exact physical center of the board
            centroid_x = np.mean(charuco_corners[:, 0, 0])
            centroid_y = np.mean(charuco_corners[:, 0, 1])
            
            # 2. Determine grid boundaries
            grid_w = image_size[0] / GRID_COLS
            grid_h = image_size[1] / GRID_ROWS
            
            # 3. Find which of the 100 buckets this centroid falls into
            col = min(int(centroid_x / grid_w), GRID_COLS - 1)
            row = min(int(centroid_y / grid_h), GRID_ROWS - 1)
            
            # 4. Check the tally
            if grid_counts[row, col] < MAX_PER_BIN:
                grid_counts[row, col] += 1
                
                # It passed the filter! Keep the data.
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                
                base_name = os.path.basename(fname)
                save_path = os.path.join(valid_folder, base_name)
                cv2.imwrite(save_path, img)

    if len(all_corners) == 0:
        print("Error: Could not find enough ChArUco corners in the images.")
        return

    print(f"\nRaw frames seen: {len(raw_imgpoints)}")
    print(f"Perfectly filtered frames kept: {len(all_corners)} (Max 5 per sector)")

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
        print("Generating 3-Panel Dashboard...")
        plot_calibration_graphs(objpoints, rvecs, tvecs, raw_imgpoints, imgpoints, image_size)

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
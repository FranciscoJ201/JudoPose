from extrinsic import calibrate_extrinsics_combined
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# Update these paths to point to your test recording files.
# ─────────────────────────────────────────────────────────────────────────
TEST_CAMERAS = {
    'realsense_0': {
        'anchor_image':  'test_data/realsense_0_anchor.jpg',
        'board_folder':  'test_data/board_images/realsense_0/',
        'intrinsic':     'test_data/realsense_0_intrinsics.json', 
        'extrinsic_out': 'test_extrinsic_rs0.json'
    },
    'realsense_1': {
        'anchor_image':  'test_data/realsense_1_anchor.jpg',
        'board_folder':  'test_data/board_images/realsense_1/',
        'intrinsic':     'test_data/realsense_1_intrinsics.json', 
        'extrinsic_out': 'test_extrinsic_rs1.json'
    }
}

MAT_SIZE_METERS = 8.0 

def plot_cameras_3d(camera_positions, mat_size):
    """
    Visualizes the Judo mat and the calculated 3D positions of the cameras.
    """
    print("\nGenerating 3D Extrinsic Dashboard...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Draw the Judo Mat (Origin to mat_size)
    xx, yy = np.meshgrid([0, mat_size], [0, mat_size])
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='green', alpha=0.2)
    
    # Draw the 4 corners of the mat
    corners = [(0,0,0), (mat_size,0,0), (mat_size,mat_size,0), (0,mat_size,0)]
    for c in corners:
        ax.scatter(*c, color='darkgreen', s=40)

    # 2. Plot the Cameras
    colors = ['red', 'blue', 'orange', 'purple']
    for i, (cam_name, pos) in enumerate(camera_positions.items()):
        x, y, z = pos
        color = colors[i % len(colors)]
        
        # Plot the camera lens
        ax.scatter(x, y, z, color=color, s=100, marker='s', label=cam_name)
        
        # Draw a dashed drop-line to the floor to visualize height
        ax.plot([x, x], [y, y], [0, z], color=color, linestyle='--', alpha=0.6)

    ax.set_xlabel('X Axis (Meters)')
    ax.set_ylabel('Y Axis (Meters)')
    ax.set_zlabel('Z Axis (Height from Floor in Meters)')
    ax.set_title('Global Camera Positions vs. Judo Mat')
    
    # Force the axes to have the same scale so the spatial relationship isn't warped
    ax.set_box_aspect([1, 1, 0.5]) 
    ax.legend()
    
    plt.show()

def run_extrinsic_test():
    print("--- Running Isolated Extrinsic Calibration Test ---")
    
    successful_cameras = {}

    for cam_name, cfg in TEST_CAMERAS.items():
        print(f"\nProcessing {cam_name}...")

        if not os.path.exists(cfg['intrinsic']):
            print(f"❌ Missing intrinsic file: {cfg['intrinsic']}")
            continue

        success = calibrate_extrinsics_combined(
            anchor_image_path  = cfg['anchor_image'],
            board_image_folder = cfg['board_folder'],
            intrinsic_json     = cfg['intrinsic'],
            output_json        = cfg['extrinsic_out'],
            mat_size_meters    = MAT_SIZE_METERS
        )

        if success and os.path.exists(cfg['extrinsic_out']):
            with open(cfg['extrinsic_out'], 'r') as f:
                ext_data = json.load(f)

            pos = ext_data['camera_position_world']
            error = ext_data['reprojection_error_px']

            print(f"✅ Success for {cam_name}!")
            print(f"   Position (X, Y, Z): ({pos[0]:.2f}m, {pos[1]:.2f}m, {pos[2]:.2f}m)")
            print(f"   Reprojection Error: {error:.3f} px")
            
            successful_cameras[cam_name] = pos
        else:
            print(f"❌ Extrinsic calibration failed for {cam_name}.")

    # If at least one camera calibrated successfully, pop open the 3D plot
    if successful_cameras:
        plot_cameras_3d(successful_cameras, MAT_SIZE_METERS)

if __name__ == "__main__":
    run_extrinsic_test()
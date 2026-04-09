from extrinsic import calibrate_extrinsics_combined
import json
import os

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# Update these paths to point to your test recording files.
# ─────────────────────────────────────────────────────────────────────────
TEST_CAMERAS = {
    'realsense_0': {
        'anchor_image':  'test_data/realsense_0_anchor.jpg',
        'board_folder':  'test_data/board_images/realsense_0/',
        'intrinsic':     'test_data/realsense_0_intrinsics.json', # From Realsense60.py
        'extrinsic_out': 'test_extrinsic_rs0.json'
    },
    'realsense_1': {
        'anchor_image':  'test_data/realsense_1_anchor.jpg',
        'board_folder':  'test_data/board_images/realsense_1/',
        'intrinsic':     'test_data/realsense_1_intrinsics.json', # From Realsense60.py
        'extrinsic_out': 'test_extrinsic_rs1.json'
    }
}

# If your test mat isn't a full 8x8 meter IJF mat, update this to the 
# actual size of the square you clicked in your anchor images.
MAT_SIZE_METERS = 8.0 

def run_extrinsic_test():
    print("--- Running Isolated Extrinsic Calibration Test ---")
    
    for cam_name, cfg in TEST_CAMERAS.items():
        print(f"\nProcessing {cam_name}...")

        # 1. Sanity check: Ensure hardware intrinsics are present
        if not os.path.exists(cfg['intrinsic']):
            print(f"❌ Missing intrinsic file: {cfg['intrinsic']}")
            continue

        # 2. Run the Extrinsic Calibration Pipeline
        success = calibrate_extrinsics_combined(
            anchor_image_path  = cfg['anchor_image'],
            board_image_folder = cfg['board_folder'],
            intrinsic_json     = cfg['intrinsic'],
            output_json        = cfg['extrinsic_out'],
            mat_size_meters    = MAT_SIZE_METERS
        )

        # 3. Read and verify the results
        if success and os.path.exists(cfg['extrinsic_out']):
            with open(cfg['extrinsic_out'], 'r') as f:
                ext_data = json.load(f)

            pos = ext_data['camera_position_world']
            error = ext_data['reprojection_error_px']

            print(f"✅ Success for {cam_name}!")
            print(f"   Position (X, Y, Z): ({pos[0]:.2f}m, {pos[1]:.2f}m, {pos[2]:.2f}m)")
            print(f"   Reprojection Error: {error:.3f} px")
        else:
            print(f"❌ Extrinsic calibration failed for {cam_name}.")

if __name__ == "__main__":
    run_extrinsic_test()
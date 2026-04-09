from basic_video_split import split_video_for_calibration
from intrinsic import calibrate_camera_charuco
from extrinsic import calibrate_extrinsics_combined   # ← updated function name
from triangulate import process_kinematics
from reid import harmonize_person_ids
from align_jsons import synchronize_and_clean_jsons
import numpy as np
import json
import os


def build_calibration_dict(cam_name, intrinsic_path, extrinsic_path):
    """
    Loads separate Intrinsic and Extrinsic JSON files and merges them
    into a single dict for downstream use in reid.py and triangulate.py.

    For the RealSense, intrinsic_path points to the hardware-exported
    realsense_intrinsics.json (from Realsense60.py), not a ChArUco result.
    The keys are different so we handle both formats here.

    Arguments:
    - cam_name (str):       Label for the camera.
    - intrinsic_path (str): Path to intrinsic JSON.
    - extrinsic_path (str): Path to extrinsic JSON.

    Returns:
    - dict with K, D, new_K, P as numpy float32 arrays.
    """
    with open(intrinsic_path, 'r') as f:
        int_data = json.load(f)

    with open(extrinsic_path, 'r') as f:
        ext_data = json.load(f)

    # Handle both ChArUco intrinsic format (camera_matrix / distortion_coefficients)
    # and RealSense hardware format (fx, fy, cx, cy / distortion_coeffs)
    if 'camera_matrix' in int_data:
        # ChArUco calibration output (GoPros)
        K = np.array(int_data['camera_matrix'],           dtype=np.float32)
        D = np.array(int_data['distortion_coefficients'], dtype=np.float32)
    else:
        # RealSense hardware intrinsics (from Realsense60.py)
        fx = int_data['fx'];  fy = int_data['fy']
        cx = int_data['cx'];  cy = int_data['cy']
        K  = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]], dtype=np.float32)
        D  = np.array(int_data['distortion_coeffs'], dtype=np.float32)

    return {
        'K':     K,
        'D':     D,
        'new_K': np.array(ext_data['optimal_camera_matrix'], dtype=np.float32),
        'P':     np.array(ext_data['projection_matrix'],     dtype=np.float32),
    }


if __name__ == "__main__":

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — CONFIGURATION
    #
    # Each camera entry needs:
    #   video            — the action recording (used for YOLO tracking)
    #   calib_folder     — folder of ChArUco frames for INTRINSIC calibration
    #                      (NOT needed for RealSense — see is_realsense flag)
    #   anchor_image     — one clean frame of the empty mat for extrinsic Phase 1
    #   board_folder     — folder of vertical ChArUco board images for extrinsic
    #                      Phase 2 (15 subfolders: position_rotation)
    #   intrinsic_out    — where to write / read the intrinsic JSON
    #   extrinsic_out    — where to write / read the extrinsic JSON
    #   is_fisheye       — True = GoPro rational model, False = standard model
    #   is_realsense     — True = skip ChArUco intrinsics, load hardware JSON instead
    #   realsense_intrinsic_json — path to realsense_intrinsics.json from Realsense60.py
    #   yolo_json        — YOLO tracking output JSON for this camera
    # ─────────────────────────────────────────────────────────────────────────

    CAMERAS = {

        'realsense': {
            'video':                    'realsense_color.mp4',
            'calib_folder':             None,           # Not needed — hardware intrinsics
            'anchor_image':             'realsense_anchor.jpg',
            'board_folder':             'board_images/realsense/',
            'intrinsic_out':            'realsense_intrinsics.json',  # Already exists from Realsense60.py
            'extrinsic_out':            'extrinsic_realsense.json',
            'is_fisheye':               False,
            'is_realsense':             True,           # ← skip ChArUco intrinsic step
            'realsense_intrinsic_json': 'realsense_intrinsics.json',
            'yolo_json':                'yolo_realsense.json',
        },

        'gopro_a': {
            'video':         'gopro_a.mp4',
            'calib_folder':  'calib_gopro_a',
            'anchor_image':  'calib_gopro_a/anchor.jpg',
            'board_folder':  'board_images/gopro_a/',
            'intrinsic_out': 'intrinsic_gopro_a.json',
            'extrinsic_out': 'extrinsic_gopro_a.json',
            'is_fisheye':    True,
            'is_realsense':  False,
            'yolo_json':     'yolo_gopro_a.json',
        },

        # 'gopro_b': {
        #     'video':         'gopro_b.mp4',
        #     'calib_folder':  'calib_gopro_b',
        #     'anchor_image':  'calib_gopro_b/anchor.jpg',
        #     'board_folder':  'board_images/gopro_b/',
        #     'intrinsic_out': 'intrinsic_gopro_b.json',
        #     'extrinsic_out': 'extrinsic_gopro_b.json',
        #     'is_fisheye':    True,
        #     'is_realsense':  False,
        #     'yolo_json':     'yolo_gopro_b.json',
        # },
    }

    MAT_SIZE_METERS = 8.0   # Standard IJF competition mat


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — SPLIT CALIBRATION VIDEOS INTO FRAMES
    # Only runs for GoPros — RealSense skips this entirely since its intrinsics
    # come from the hardware export in Realsense60.py.
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- STEP 2: Splitting calibration videos ---")
    for cam_name, cfg in CAMERAS.items():
        if cfg['is_realsense']:
            print(f"  {cam_name}: skipping (hardware intrinsics from Realsense60.py)")
            continue
        split_video_for_calibration(cfg['video'], cfg['calib_folder'], frame_skip=1)


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — INTRINSIC CALIBRATION
    # Only runs for GoPros. RealSense intrinsics already exist as
    # realsense_intrinsics.json written by Realsense60.py at recording time.
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- STEP 3: Intrinsic calibration ---")
    for cam_name, cfg in CAMERAS.items():
        if cfg['is_realsense']:
            print(f"  {cam_name}: skipping (using {cfg['realsense_intrinsic_json']})")
            continue
        calibrate_camera_charuco(
            image_folder=cfg['calib_folder'],
            output_file=cfg['intrinsic_out'],
            is_fisheye=cfg['is_fisheye'],
            show_graph=True
        )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — EXTRINSIC CALIBRATION (per camera)
    #
    # Uses the new combined Option 1 + Option 3 approach:
    #   Phase 1 — click 4 mat corners to anchor world coordinate system
    #   Phase 2 — detect vertical ChArUco board at 5 positions × 3 rotations
    #   Phase 3 — combined solvePnP on 300+ points with reprojection error check
    #
    # The function returns True on success and False if reprojection error
    # exceeds the 1.5px threshold. The pipeline aborts on any failure so
    # you don't silently proceed with a bad calibration.
    #
    # Board folder structure required per camera:
    #   board_images/{cam_name}/
    #       top_left_facing_x/
    #       top_left_facing_y/
    #       top_left_diagonal/
    #       top_right_facing_x/
    #       ... (15 subfolders total)
    #       centre_diagonal/
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- STEP 4: Extrinsic calibration ---")
    for cam_name, cfg in CAMERAS.items():
        print(f"\n  Camera: {cam_name}")
        success = calibrate_extrinsics_combined(
            anchor_image_path  = cfg['anchor_image'],
            board_image_folder = cfg['board_folder'],
            intrinsic_json     = cfg['intrinsic_out'],
            output_json        = cfg['extrinsic_out'],
            mat_size_meters    = MAT_SIZE_METERS
        )
        if not success:
            # A failed extrinsic means every 3D point from this camera will be
            # wrong. Hard stop here rather than silently producing bad output.
            raise RuntimeError(
                f"Extrinsic calibration failed for {cam_name}. "
                f"Check anchor image, board images, and square size measurement. "
                f"See printed diagnostics above."
            )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — BUILD MASTER CALIBRATION DICT
    # Merges intrinsic + extrinsic data into one structure.
    # Handles both ChArUco and RealSense hardware intrinsic formats.
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- STEP 5: Building master calibration dict ---")
    master_calibration = {}
    for cam_name, cfg in CAMERAS.items():
        master_calibration[cam_name] = build_calibration_dict(
            cam_name,
            cfg['intrinsic_out'],
            cfg['extrinsic_out']
        )
        cam_pos = json.load(open(cfg['extrinsic_out']))['camera_position_world']
        print(f"  {cam_name}: position X={cam_pos[0]:.2f}m  "
              f"Y={cam_pos[1]:.2f}m  Z={cam_pos[2]:.2f}m  "
              f"| reproj error: "
              f"{json.load(open(cfg['extrinsic_out']))['reprojection_error_px']:.3f}px")


# ─────────────────────────────────────────────────────────────────────────
    # STEP 5.5 — TIMELINE SYNCHRONIZATION & FRAME DROPPING (Stage 4b)
    # Re-aligns the hardware timelines using linear regression and deletes
    # the USB blackout frames from all cameras.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- STEP 5.5: Timeline Synchronization ---")
    
    csv_paths = {
        'realsense': 'realsense_cam_1/timestamps.csv', # Update paths accordingly
        'gopro_a': 'realsense_cam_0/timestamps.csv'    # Update paths accordingly
    }
    
    raw_yolo_paths = {
        cam_name: cfg['yolo_json']
        for cam_name, cfg in CAMERAS.items()
    }
    
    # This outputs new paths to the cleaned JSONs
    synced_json_paths = synchronize_and_clean_jsons(
        master_cam='realsense', # Designate your flawless camera here
        camera_csv_paths=csv_paths,
        camera_json_paths=raw_yolo_paths,
        output_dir='synced_yolo_output',
        threshold_ms=12.0
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — RE-ID: HARMONIZE PERSON IDs ACROSS CAMERAS
    # Ensures detections[0] = Athlete A and detections[1] = Athlete B
    # consistently across every camera and every frame.
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- STEP 6: Harmonizing person IDs across cameras ---")

    raw_yolo_jsons = {
        cam_name: cfg['yolo_json']
        for cam_name, cfg in CAMERAS.items()
    }

    video_paths = {
        cam_name: cfg['video']
        for cam_name, cfg in CAMERAS.items()
    }

    harmonized_paths = harmonize_person_ids(
        camera_json_paths  = synced_json_paths,
        calibration_dict   = master_calibration,
        output_dir         = 'harmonized_output',
        video_paths        = video_paths,
        reverify_interval  = 30
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — 3D TRIANGULATION
    # Feeds harmonized JSONs into the weighted DLT solver.
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- STEP 7: 3D Triangulation ---")

    process_kinematics(
        camera_json_paths    = harmonized_paths,
        calibration_dict     = master_calibration,
        output_path          = "final_judo_throw_3d.json",
        confidence_threshold = 0.5,
        min_cameras          = 2,
        do_optimization      = False
    )
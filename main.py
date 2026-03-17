from video_split import split_video_for_calibration
from intrinsic import calibrate_camera_charuco
from extrinsic import calibrate_extrinsics_pnp
from triangulate import process_kinematics
from reid import harmonize_person_ids          
import numpy as np
import json
import os

def build_calibration_dict(cam_name, intrinsic_path, extrinsic_path):
    """
    Loads separate Intrinsic and Extrinsic JSON files and merges them.

    Arguments:
    - cam_name (str): Label for the camera.
    - intrinsic_path (str): File path to the camera's intrinsic JSON.
    - extrinsic_path (str): File path to the camera's extrinsic JSON.

    Returns:
    - dict: Contains K, D, new_K, and P as numpy arrays.
    """
    with open(intrinsic_path, 'r') as f:
        int_data = json.load(f)

    with open(extrinsic_path, 'r') as f:
        ext_data = json.load(f)

    return {
        'K':     np.array(int_data['camera_matrix'],          dtype=np.float32),
        'D':     np.array(int_data['distortion_coefficients'], dtype=np.float32),
        'new_K': np.array(ext_data['optimal_camera_matrix'],  dtype=np.float32),
        'P':     np.array(ext_data['projection_matrix'],      dtype=np.float32)
    }


if __name__ == "__main__":

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — CONFIGURATION
    # Add or remove cameras here. Every camera needs a video, two intrinsic
    # calibration images, and a clear view of the mat for extrinsics.
    # ─────────────────────────────────────────────────────────────────────────

    CAMERAS = {
        'cam_a': {
            'video':          'test4.mp4',
            'calib_folder':   'image_folder1',
            'intrinsic_out':  'intrinsic_cam_a.json',
            'extrinsic_out':  'extrinsic_cam_a.json',
            'is_fisheye':     True,    # True = Rational Model (GoPro). False = Standard (RealSense)
            'yolo_json':      'yolo_cam_a.json',   # Output from your YOLOv8 model.track() step
        },
        # 'cam_b': {
        #     'video':          'lineartest.mp4',
        #     'calib_folder':   'image_folder2',
        #     'intrinsic_out':  'intrinsic_cam_b.json',
        #     'extrinsic_out':  'extrinsic_cam_b.json',
        #     'is_fisheye':     False,
        #     'yolo_json':      'yolo_cam_b.json',
        # },
        # 'cam_c': {
        #     'video':          'cam_c.mp4',
        #     'calib_folder':   'image_folder3',
        #     'intrinsic_out':  'intrinsic_cam_c.json',
        #     'extrinsic_out':  'extrinsic_cam_c.json',
        #     'is_fisheye':     False,
        #     'yolo_json':      'yolo_cam_c.json',
        # },
    }

    MAT_SIZE_METERS = 8.0   # Standard IJF competition mat
    EXTRINSIC_FRAME = 'calib_0085.jpg'  # A clear frame from each calib folder


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — SPLIT CALIBRATION VIDEOS INTO FRAMES
    # Extracts JPEGs from each camera's video for ChArUco detection.
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- STEP 2: Splitting calibration videos ---")
    for cam_name, cfg in CAMERAS.items():
        split_video_for_calibration(cfg['video'], cfg['calib_folder'], frame_skip=1)


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — INTRINSIC CALIBRATION (per camera)
    # Fits lens distortion model from ChArUco board images.
    # is_fisheye=True uses the Rational Model (8 distortion coefficients).
    # is_fisheye=False uses the Standard Model (5 coefficients).
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- STEP 3: Intrinsic calibration ---")
    for cam_name, cfg in CAMERAS.items():
        calibrate_camera_charuco(
            image_folder=cfg['calib_folder'],
            output_file=cfg['intrinsic_out'],
            is_fisheye=cfg['is_fisheye'],
            show_graph=True
        )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — EXTRINSIC CALIBRATION (per camera)
    # Calculates each camera's 3D position by clicking the 4 mat corners.
    # You will be shown a window — click corners in order: TL, TR, BR, BL.
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- STEP 4: Extrinsic calibration (click mat corners when prompted) ---")
    for cam_name, cfg in CAMERAS.items():
        frame_path = os.path.join(cfg['calib_folder'], EXTRINSIC_FRAME)
        calibrate_extrinsics_pnp(
            image_path=frame_path,
            intrinsic_json=cfg['intrinsic_out'],
            output_json=cfg['extrinsic_out'],
            mat_size_meters=MAT_SIZE_METERS
        )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — BUILD MASTER CALIBRATION DICT
    # Merges intrinsic + extrinsic data into one structure for downstream steps.
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- STEP 5: Building master calibration dict ---")
    master_calibration = {
        cam_name: build_calibration_dict(cam_name, cfg['intrinsic_out'], cfg['extrinsic_out'])
        for cam_name, cfg in CAMERAS.items()
    }


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — RE-ID: HARMONIZE PERSON IDs ACROSS CAMERAS        ← NEW STEP
    # Ensures detections[0] = Athlete A and detections[1] = Athlete B
    # consistently across every camera and every frame.
    # Requires: YOLO was run with model.track() (not model.predict()).
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- STEP 6: Harmonizing person IDs across cameras ---")

    raw_yolo_jsons = {
        cam_name: cfg['yolo_json']
        for cam_name, cfg in CAMERAS.items()
    }

    # Optional: pass video_paths to enable colour-based tiebreaking when
    # two athletes are very close together and geometry alone is ambiguous.
    video_paths = {
        cam_name: cfg['video']
        for cam_name, cfg in CAMERAS.items()
    }

    harmonized_paths = harmonize_person_ids(
        camera_json_paths=raw_yolo_jsons,
        calibration_dict=master_calibration,
        output_dir='harmonized_output',
        video_paths=video_paths,    # Remove this line to skip colour fallback
        reverify_interval=30        # Re-check cross-camera mapping every 30 frames
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — 3D TRIANGULATION
    # Feeds the harmonized (ID-stable) JSONs into the DLT solver.
    # Each frame now produces a reliable 3D skeleton for both athletes.
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- STEP 7: 3D Triangulation ---")

    process_kinematics(
        camera_json_paths=harmonized_paths,     # ← harmonized, not raw
        calibration_dict=master_calibration,
        output_path="final_judo_throw_3d.json",
        confidence_threshold=0.5,
        min_cameras=2,
        do_optimization=False                   # Toggle True when polish_3d_point is implemented
    )
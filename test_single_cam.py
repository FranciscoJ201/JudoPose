import numpy as np
import json
import os

# Import the engines we just built
from triangulate import batched_depth_deproject
from smpl_fit import SMPLFitter

def test_single_realsense_smpl(yolo_json_path, depth_npy_path, intrinsics_json_path, smpl_model_path, frame_index=150, output_npz="test_single_kinematics.npz"):
    """
    Isolates a single RealSense camera to test depth deprojection and SMPL fitting.
    Bypasses all multi-camera synchronization and global mat extrinsics.

    Arguments:
    - yolo_json_path:       (str) Path to the raw YOLO tracking JSON.
    - depth_npy_path:       (str) Path to the exact .npy depth frame matching the frame_index.
    - intrinsics_json_path: (str) Path to the realsense_intrinsics.json from the capture script.
    - smpl_model_path:      (str) Path to the folder containing SMPL_NEUTRAL.pkl.
    - frame_index:          (int) The specific frame number to test.
    - output_npz:           (str) Where to save the resulting single-frame mesh data.
    """
    
    print(f"[Single Test] Loading Frame {frame_index} data...")
    
    # 1. Load Hardware Intrinsics
    with open(intrinsics_json_path, 'r') as f:
        intr = json.load(f)
        
    K = np.array([
        [intr['fx'], 0,          intr['cx']],
        [0,          intr['fy'], intr['cy']],
        [0,          0,          1         ]
    ], dtype=np.float32)

    # 2. Load YOLO 2D Coordinates
    with open(yolo_json_path, 'r') as f:
        yolo_data = json.load(f)
        
    frame_data = yolo_data[frame_index]
    detections = frame_data.get('detections', [])
    
    if not detections:
        print("[Single Test] Error: No person detected in this frame.")
        return
        
    # Extract the 17 (u,v) coordinates
    kpts_2d = np.array(detections[0].get('keypoints_2d', detections[0].get('keypoints_3d_m', [])))
    uv_coords = kpts_2d[:, :2].astype(np.float32)

    # 3. Load the raw .npy depth map
    depth_map = np.load(depth_npy_path)

    # 4. Local Deprojection (Bypassing Extrinsics)
    print("[Single Test] Deprojecting 2D pixels to Local 3D Space...")
    R_identity = np.eye(3, dtype=np.float32)       # No rotation
    t_zero = np.zeros((3, 1), dtype=np.float32)    # No translation
    
    local_3d_pts = batched_depth_deproject(
        uv_coords=uv_coords,
        depth_map=depth_map,
        new_K=K, 
        R=R_identity, 
        t=t_zero,
        window_size=5
    )

    # 5. Fit the SMPL Model
    print("[Single Test] Fitting SMPL Biomechanical Mesh...")
    fitter = SMPLFitter(smpl_model_path=smpl_model_path)
    
    pose, shape, trans = fitter.fit_frame(
        target_3d_joints=local_3d_pts,
        iterations=150 # Higher iterations for a single frame cold-start
    )

    # 6. Save for Visualization
    np.savez(
        output_npz, 
        poses=pose.cpu().numpy(), 
        shapes=shape.cpu().numpy(), 
        trans=trans.cpu().numpy()
    )
    
    print(f"[Single Test] Success! Test data saved to {output_npz}")
    print("[Single Test] Run visualize_smpl.py on this file to see the mesh.")

if __name__ == "__main__":
    # Example execution: Update these paths to your local test files
    """
    test_single_realsense_smpl(
        yolo_json_path="yolo_realsense_0.json",
        depth_npy_path="realsense_cam_0/depth_frames/depth_000150.npy",
        intrinsics_json_path="realsense_cam_0/realsense_intrinsics.json",
        smpl_model_path="models/",
        frame_index=150
    )
    """
    pass
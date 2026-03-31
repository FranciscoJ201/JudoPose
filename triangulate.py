import numpy as np
import json
import cv2
import os
from scipy.interpolate import interp1d
from scipy.optimize import least_squares


def _build_dlt_A_matrix(P_stack, uv_stack, weight_stack):
    """
    Builds the weighted DLT A matrix for ALL joints in one frame simultaneously.

    Instead of looping over joints and calling this 17 times, we construct A
    for all joints at once using broadcasting.

    Arguments:
    - P_stack:      (num_valid_cams, 3, 4)  — projection matrices for cameras that passed threshold
    - uv_stack:     (num_valid_cams, num_joints, 2) — undistorted pixel coords
    - weight_stack: (num_valid_cams, num_joints)    — YOLO confidence scores

    Returns:
    - A: (num_joints, 2*num_valid_cams, 4) — one DLT system per joint, stacked
    """
    num_cams, num_joints, _ = uv_stack.shape

    # u: (num_cams, num_joints), v: (num_cams, num_joints)
    u = uv_stack[:, :, 0]
    v = uv_stack[:, :, 1]

    # P rows: P2 = P[:, 2, :] shape (num_cams, 4)
    # Broadcast over joints: (num_cams, 1, 4) - (num_cams, 1, 4)
    P0 = P_stack[:, 0, :][:, None, :]   
    P1 = P_stack[:, 1, :][:, None, :]   
    P2 = P_stack[:, 2, :][:, None, :]   

    # row1[cam, joint] = weight * (u * P2 - P0), shape (num_cams, num_joints, 4)
    row1 = (weight_stack[:, :, None] *
            (u[:, :, None] * P2 - P0))

    # row2[cam, joint] = weight * (v * P2 - P1), shape (num_cams, num_joints, 4)
    row2 = (weight_stack[:, :, None] *
            (v[:, :, None] * P2 - P1))

    # Interleave row1 and row2: (num_joints, 2*num_cams, 4)
    # Transpose to (num_joints, num_cams, 4) first, then interleave
    r1 = row1.transpose(1, 0, 2)   # (num_joints, num_cams, 4)
    r2 = row2.transpose(1, 0, 2)   # (num_joints, num_cams, 4)

    # Stack along axis=1: (num_joints, 2*num_cams, 4)
    A = np.empty((num_joints, 2 * num_cams, 4), dtype=np.float32)
    A[:, 0::2, :] = r1
    A[:, 1::2, :] = r2

    return A


def batched_dlt_triangulate(P_stack, uv_stack, weight_stack):
    """
    Triangulates ALL joints in a single frame in one batched SVD call.

    Arguments:
    - P_stack:      (num_valid_cams, 3, 4)
    - uv_stack:     (num_valid_cams, num_joints, 2)
    - weight_stack: (num_valid_cams, num_joints)

    Returns:
    - points_3d: (num_joints, 3) — triangulated 3D coordinates
    """
    A = _build_dlt_A_matrix(P_stack, uv_stack, weight_stack)
    num_joints = A.shape[0]
    points_3d = np.full((num_joints, 3), np.nan)
    
    # THE GATEKEEPER: Only pass matrices to SVD that have NO NaNs
    valid_mask = ~np.isnan(A).any(axis=(1, 2))
    A_valid = A[valid_mask]

    if len(A_valid) == 0:
        return points_3d

    # np.linalg.svd accepts batched input
    # Returns Vh: (N, 4, 4) — last row of each is the solution
    _, _, Vh = np.linalg.svd(A_valid)

    # Last row of Vh for each joint: (N, 4)
    X_homo = Vh[:, -1, :]

    # Dehomogenise: divide XYZ by W
    W = X_homo[:, 3:4]
    
    # Avoid division by zero
    nonzero_mask = (np.abs(W.flatten()) > 1e-6)
    
    X_3d = np.full((len(A_valid), 3), np.nan)
    X_3d[nonzero_mask] = X_homo[nonzero_mask, :3] / W[nonzero_mask]

    points_3d[valid_mask] = X_3d
    return points_3d


def batched_depth_deproject(uv_coords, depth_map, new_K, R, t, window_size=5):
    """
    Extracts 3D global coordinates from a single RealSense depth map when DLT fails.
    Extracts a window around the joint, filters dead pixels, and applies transformations.

    Arguments:
    - uv_coords:   (N, 2) numpy array of (u, v) pixel coordinates.
    - depth_map:   (H, W) numpy uint16 array of raw depth in millimeters.
    - new_K:       (3, 3) Optimal camera intrinsic matrix (matches undistorted UVs).
    - R:           (3, 3) Extrinsic rotation matrix.
    - t:           (3, 1) Extrinsic translation vector.
    - window_size: (int) Size of the median filter window (must be odd).

    Returns:
    - P_global:    (N, 3) numpy array of [X, Y, Z] coordinates in meters.
    """
    num_joints = uv_coords.shape[0]
    H, W = depth_map.shape
    half_w = window_size // 2

    P_global = np.full((num_joints, 3), np.nan)

    fx, fy = new_K[0, 0], new_K[1, 1]
    cx, cy = new_K[0, 2], new_K[1, 2]
    R_inv = R.T

    for i in range(num_joints):
        if np.isnan(uv_coords[i, 0]):
            continue

        u, v = int(np.round(uv_coords[i, 0])), int(np.round(uv_coords[i, 1]))

        # Boundary checks to prevent indexing crashes
        u_min, u_max = max(0, u - half_w), min(W, u + half_w + 1)
        v_min, v_max = max(0, v - half_w), min(H, v + half_w + 1)

        if u_min >= u_max or v_min >= v_max:
            continue

        patch = depth_map[v_min:v_max, u_min:u_max].astype(float)
        patch[patch == 0] = np.nan # Ignore dead IR pixels (zeros)

        if not np.all(np.isnan(patch)):
            Z_local = np.nanmedian(patch) / 1000.0 # Convert mm to meters

            # 1. Local Deprojection
            X_local = (uv_coords[i, 0] - cx) * Z_local / fx
            Y_local = (uv_coords[i, 1] - cy) * Z_local / fy
            P_local = np.array([X_local, Y_local, Z_local]).reshape(3, 1)

            # 2. Global Transformation
            P_g = R_inv @ (P_local - t)
            P_global[i] = P_g.flatten()

    return P_global


def reprojection_residuals(pt_3d, P_matrices, uv_targets, conf_weights):
    """
    Calculates the error between the mathematically projected 3D point 
    and the actual 2D YOLO detections.
    """
    # 1. Convert 3D point to homogeneous coordinates [X, Y, Z, 1]
    pt_3d_homo = np.append(pt_3d, 1.0) 

    # 2. Project the 3D point back onto all 2D camera planes simultaneously
    projected_homo = P_matrices @ pt_3d_homo

    # 3. Dehomogenize to get standard (u, v) pixel coordinates
    u_proj = projected_homo[:, 0] / projected_homo[:, 2]
    v_proj = projected_homo[:, 1] / projected_homo[:, 2]
    uv_proj = np.column_stack((u_proj, v_proj)) 

    # 4. Calculate the raw pixel distance (error)
    residuals = uv_proj - uv_targets 

    # 5. Multiply the error by the YOLO confidence score
    weighted_residuals = residuals * conf_weights[:, None]

    # least_squares expects a flat 1D array of errors
    return weighted_residuals.flatten()


def polish_3d_point(initial_3d_point, P_matrices, uv_targets, conf_weights):
    """
    Applies Levenberg-Marquardt non-linear optimization to micro-adjust 
    a 3D coordinate until the multi-camera reprojection error is minimized.
    """
    if np.isnan(initial_3d_point).any():
        return initial_3d_point

    # Run the Levenberg-Marquardt optimizer
    result = least_squares(
        fun=reprojection_residuals,
        x0=initial_3d_point,
        args=(P_matrices, uv_targets, conf_weights),
        method='lm',     # The Levenberg-Marquardt algorithm
        max_nfev=50      # Cap iterations to keep the pipeline fast
    )

    if result.success:
        return result.x
    else:
        # If the math fails to converge, fallback to the original DLT guess
        return initial_3d_point


def interpolate_missing_data(trajectory_3d):
    """
    Fills NaNs across a joint's timeline using cubic spline interpolation.
    """
    frames = np.arange(len(trajectory_3d))
    valid_mask = ~np.isnan(trajectory_3d[:, 0])

    if not np.any(valid_mask):
        return trajectory_3d

    valid_frames = frames[valid_mask]
    valid_data   = trajectory_3d[valid_mask, :]      # (num_valid, 3)

    interpolator = interp1d(
        valid_frames, valid_data,
        kind='cubic', axis=0, fill_value='extrapolate'
    )

    return interpolator(frames)


def process_kinematics(camera_json_paths, calibration_dict, output_path, depth_folders=None,
                       confidence_threshold=0.6, min_cameras=2, do_optimization=False):
    """
    Main Hybrid Triangulation pipeline. 
    Routes multi-camera joints to DLT, and single-camera joints to Depth Deprojection.
    """
    print("Loading YOLO tracking data...")
    cam_data = {}
    for cam_name, path in camera_json_paths.items():
        with open(path, 'r') as f:
            cam_data[cam_name] = json.load(f)

    num_frames    = min(len(data) for data in cam_data.values())
    num_joints    = 17
    cam_names     = list(camera_json_paths.keys())
    num_cams      = len(cam_names)

    final_3d_timeline = np.full((num_frames, num_joints, 3), np.nan)

    print(f"Processing {num_frames} frames × {num_joints} joints × {num_cams} cameras...")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — SCRAPE JSON + BATCH UNDISTORT
    # ─────────────────────────────────────────────────────────────────────────
    print("Pre-computing batched undistortion...")

    all_flat   = np.full((num_cams, num_frames, num_joints, 2), np.nan, dtype=np.float32)
    all_conf   = np.zeros((num_cams, num_frames, num_joints),            dtype=np.float32)
    all_valid  = np.zeros((num_cams, num_frames, num_joints),            dtype=bool)

    for cam_idx, cam_name in enumerate(cam_names):
        raw_pts = np.full((num_frames * num_joints, 1, 2), np.nan, dtype=np.float32)
        conf_arr  = np.zeros((num_frames, num_joints), dtype=np.float32)
        valid_arr = np.zeros((num_frames, num_joints), dtype=bool)

        for frame_idx in range(num_frames):
            try:
                person_data = cam_data[cam_name][frame_idx]['detections'][0]
                kps = person_data.get('keypoints_2d', person_data.get('keypoints_3d_m', []))

                for joint_idx in range(min(num_joints, len(kps))):
                    kp = kps[joint_idx]
                    if kp[0] is not None and kp[1] is not None:
                        flat_idx = frame_idx * num_joints + joint_idx
                        raw_pts[flat_idx, 0, 0] = float(kp[0])
                        raw_pts[flat_idx, 0, 1] = float(kp[1])
                        conf_arr[frame_idx, joint_idx]  = float(kp[2])
                        valid_arr[frame_idx, joint_idx] = True

            except (IndexError, KeyError):
                continue

        K     = calibration_dict[cam_name]['K']
        D     = calibration_dict[cam_name]['D']
        new_K = calibration_dict[cam_name]['new_K']

        safe_pts = np.where(np.isnan(raw_pts), 0.0, raw_pts)
        flat_pts = cv2.undistortPoints(safe_pts, K, D, None, new_K)
        flat_reshaped = flat_pts.reshape((num_frames, num_joints, 2))
        flat_reshaped[~valid_arr] = 0.0

        all_flat[cam_idx]  = flat_reshaped
        all_conf[cam_idx]  = conf_arr
        all_valid[cam_idx] = valid_arr

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — HYBRID TRIANGULATION (DLT + DEPROJECTION)
    # ─────────────────────────────────────────────────────────────────────────
    print("Running Hybrid Triangulation (DLT + Depth Fallback)...")

    P_all = np.stack([calibration_dict[c]['P'] for c in cam_names], axis=0)

    for frame_idx in range(num_frames):

        frame_conf  = all_conf[:, frame_idx, :]   
        frame_valid = all_valid[:, frame_idx, :]  
        frame_uv    = all_flat[:, frame_idx, :]   

        usable = frame_valid & (frame_conf >= confidence_threshold) 
        cam_usable_any = usable.any(axis=1)  
        valid_cam_indices = np.where(cam_usable_any)[0]

        if len(valid_cam_indices) == 0:
            continue

        P_frame    = P_all[valid_cam_indices]           
        uv_frame   = frame_uv[valid_cam_indices]        
        conf_frame = frame_conf[valid_cam_indices]      
        use_frame  = usable[valid_cam_indices]          

        # ─── SCENARIO A: DLT (>= 2 Cameras) ───
        conf_frame_safe = np.where(use_frame, conf_frame, 0.0)
        pts_3d = batched_dlt_triangulate(P_frame, uv_frame, conf_frame_safe)
        
        joint_cam_counts = use_frame.sum(axis=0)  
        sufficient = joint_cam_counts >= min_cameras
        final_3d_timeline[frame_idx, sufficient] = pts_3d[sufficient]

        # ─── SCENARIO B: DEPROJECTION (1 Camera Fallback) ───
        if depth_folders is not None:
            deproject_mask = (joint_cam_counts == 1)
            deproject_joints = np.where(deproject_mask)[0]
            
            if len(deproject_joints) > 0:
                loaded_depth_maps = {} # Cache to prevent reading the same .npy file multiple times
                
                for j in deproject_joints:
                    # Find which camera in the valid subset sees this specific joint
                    c_idx_local = np.where(use_frame[:, j])[0][0]
                    c_idx_global = valid_cam_indices[c_idx_local]
                    cam_name = cam_names[c_idx_global]
                    
                    if cam_name in depth_folders:
                        if cam_name not in loaded_depth_maps:
                            # Align hardware indices with synced timeline indices
                            orig_hw_idx = cam_data[cam_name][frame_idx].get("orig_hardware_index", frame_idx)
                            depth_path = os.path.join(depth_folders[cam_name], f"depth_{orig_hw_idx:06d}.npy")
                            
                            if os.path.exists(depth_path):
                                loaded_depth_maps[cam_name] = np.load(depth_path)
                            else:
                                loaded_depth_maps[cam_name] = None
                                
                        dmap = loaded_depth_maps[cam_name]
                        if dmap is not None:
                            new_K = calibration_dict[cam_name]['new_K']
                            R = calibration_dict[cam_name]['R']
                            t = calibration_dict[cam_name]['t']
                            
                            # Pass only the specific joint to the deprojector
                            pt_3d = batched_depth_deproject(
                                uv_frame[c_idx_local, j:j+1, :], 
                                dmap, new_K, R, t
                            )
                            final_3d_timeline[frame_idx, j] = pt_3d[0]


        if do_optimization:
            # Only polish joints that were triangulated using DLT (Scenario A)
            for j in np.where(sufficient)[0]:
                final_3d_timeline[frame_idx, j] = polish_3d_point(
                    initial_3d_point=final_3d_timeline[frame_idx, j], 
                    P_matrices=P_frame, 
                    uv_targets=uv_frame[:, j, :],
                    conf_weights=conf_frame_safe[:, j] # NEW: Pass the YOLO weights
                )

        if frame_idx > 0 and frame_idx % 300 == 0:
            print(f"  Frame {frame_idx}/{num_frames} — "
                  f"{sufficient.sum()} DLT, {(joint_cam_counts == 1).sum()} Deprojected")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 — INTERPOLATION
    # ─────────────────────────────────────────────────────────────────────────
    print("Running cubic spline interpolation to fix remaining occlusions...")
    for joint_idx in range(num_joints):
        final_3d_timeline[:, joint_idx, :] = interpolate_missing_data(
            final_3d_timeline[:, joint_idx, :]
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4 — SAVE
    # ─────────────────────────────────────────────────────────────────────────
    print("Saving final 3D skeleton to JSON...")

    was_nan = np.isnan(final_3d_timeline[:, :, 0])  

    output_data = []
    for frame_idx in range(num_frames):
        keypoints = []
        for joint_idx in range(num_joints):
            keypoints.append({
                "xyz":          final_3d_timeline[frame_idx, joint_idx].tolist(),
                "interpolated": bool(was_nan[frame_idx, joint_idx])
            })
        output_data.append({
            "frame_index": frame_idx,
            "keypoints_3d": keypoints
        })

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    pass
import numpy as np
import json
import cv2
from scipy.interpolate import interp1d


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
    P0 = P_stack[:, 0, :]   # (num_cams, 4)
    P1 = P_stack[:, 1, :]   # (num_cams, 4)
    P2 = P_stack[:, 2, :]   # (num_cams, 4)

    # row1[cam, joint] = weight * (u * P2 - P0), shape (num_cams, num_joints, 4)
    row1 = (weight_stack[:, :, None] *
            (u[:, :, None] * P2[:, None, :] - P0[:, None, :]))

    # row2[cam, joint] = weight * (v * P2 - P1), shape (num_cams, num_joints, 4)
    row2 = (weight_stack[:, :, None] *
            (v[:, :, None] * P2[:, None, :] - P1[:, None, :]))

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

    # np.linalg.svd accepts batched input (num_joints, 2*num_cams, 4)
    # Returns Vh: (num_joints, 4, 4) — last row of each is the solution
    _, _, Vh = np.linalg.svd(A)

    # Last row of Vh for each joint: (num_joints, 4)
    X = Vh[:, -1, :]

    # Dehomogenise: divide XYZ by W
    return X[:, :3] / X[:, 3:4]


def polish_3d_point(initial_3d_point, projection_matrices, points_2d):
    """
    TODO: Non-Linear Optimization Step (Levenberg-Marquardt).

    Arguments:
    - initial_3d_point: (num_joints, 3) array from batched DLT.
    - projection_matrices: List of 3x4 P matrices.
    - points_2d: (num_cams, num_joints, 2) undistorted pixel coords.
    """
    return initial_3d_point


def interpolate_missing_data(trajectory_3d):
    """
    Fills NaNs across a joint's timeline using cubic spline interpolation.
    Operates on all 3 axes simultaneously via scipy's interp1d.

    Arguments:
    - trajectory_3d: (num_frames, 3)

    Returns:
    - (num_frames, 3) with NaNs filled.
    """
    frames = np.arange(len(trajectory_3d))
    valid_mask = ~np.isnan(trajectory_3d[:, 0])

    if not np.any(valid_mask):
        return trajectory_3d

    valid_frames = frames[valid_mask]
    valid_data   = trajectory_3d[valid_mask, :]      # (num_valid, 3)

    # interp1d with axis=0 handles all 3 axes in one call — no axis loop needed
    interpolator = interp1d(
        valid_frames, valid_data,
        kind='cubic', axis=0, fill_value='extrapolate'
    )

    return interpolator(frames)


def process_kinematics(camera_json_paths, calibration_dict, output_path,
                       confidence_threshold=0.6, min_cameras=2, do_optimization=False):
    """
    Main pipeline function. Loads YOLO JSONs, batches undistortion once per camera,
    then triangulates all joints per frame in a single SVD call instead of 17.

    Arguments:
    - camera_json_paths (dict): cam_name → path to YOLO tracking JSON.
    - calibration_dict (dict):  cam_name → {K, D, new_K, P} numpy arrays.
    - output_path (str):        Where to write the final 3D skeleton JSON.
    - confidence_threshold (float): Minimum YOLO confidence to use a camera ray.
    - min_cameras (int):        Minimum cameras needed to attempt triangulation.
    - do_optimization (bool):   Toggle for the future polish_3d_point step.
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
    # PHASE 1 — SCRAPE JSON + BATCH UNDISTORT (one OpenCV call per camera)
    #
    # raw_pts: (num_cams, num_frames*num_joints, 1, 2) — OpenCV batched format
    # conf:    (num_cams, num_frames, num_joints)
    # valid:   (num_cams, num_frames, num_joints) bool — False where YOLO had no detection
    #
    # The key fix vs previous version: NaN-init raw_pts so missing detections
    # don't silently produce (0,0) which is a valid top-left pixel coordinate.
    # We track a separate boolean validity mask and gate on BOTH conf AND valid.
    # ─────────────────────────────────────────────────────────────────────────
    print("Pre-computing batched undistortion...")

    all_flat   = np.full((num_cams, num_frames, num_joints, 2), np.nan, dtype=np.float32)
    all_conf   = np.zeros((num_cams, num_frames, num_joints),            dtype=np.float32)
    all_valid  = np.zeros((num_cams, num_frames, num_joints),            dtype=bool)

    for cam_idx, cam_name in enumerate(cam_names):
        # Allocate NaN-initialised raw points — missing joints stay NaN,
        # never silently default to pixel (0, 0)
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

        # Replace NaN slots with (0,0) temporarily — OpenCV can't handle NaN.
        # We'll gate them out with valid_arr so they never reach the DLT.
        safe_pts = np.where(np.isnan(raw_pts), 0.0, raw_pts)

        # Single batched undistort call for the entire camera timeline
        flat_pts = cv2.undistortPoints(safe_pts, K, D, None, new_K)

        # Reshape to (num_frames, num_joints, 2)
        flat_reshaped = flat_pts.reshape((num_frames, num_joints, 2))

        # Zero out any slots that were invalid — they should never be read,
        # but this makes bugs obvious (a real coordinate wouldn't be exactly 0,0)
        flat_reshaped[~valid_arr] = 0.0

        all_flat[cam_idx]  = flat_reshaped
        all_conf[cam_idx]  = conf_arr
        all_valid[cam_idx] = valid_arr

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — BATCHED DLT TRIANGULATION (one SVD call per frame, not per joint)
    #
    # For each frame we:
    # 1. Find which cameras cleared threshold for enough joints
    # 2. Build the full A matrix for all 17 joints at once
    # 3. Run one np.linalg.svd call that returns all 17 solutions simultaneously
    # ─────────────────────────────────────────────────────────────────────────
    print("Running batched DLT triangulation...")

    # Stack P matrices: (num_cams, 3, 4)
    P_all = np.stack([calibration_dict[c]['P'] for c in cam_names], axis=0)

    for frame_idx in range(num_frames):

        # conf: (num_cams, num_joints), valid: (num_cams, num_joints)
        frame_conf  = all_conf[:, frame_idx, :]   # (num_cams, num_joints)
        frame_valid = all_valid[:, frame_idx, :]  # (num_cams, num_joints)
        frame_uv    = all_flat[:, frame_idx, :]   # (num_cams, num_joints, 2)

        # A camera is usable for a joint if it has a valid detection AND confidence passes
        usable = frame_valid & (frame_conf >= confidence_threshold)  # (num_cams, num_joints)

        # Find cameras that are usable for at least one joint this frame
        # We triangulate per joint only when enough cameras see it
        # For efficiency: find the set of cameras usable across the whole frame
        cam_usable_any = usable.any(axis=1)  # (num_cams,) — cams with any valid joint

        valid_cam_indices = np.where(cam_usable_any)[0]

        if len(valid_cam_indices) < min_cameras:
            # Not enough cameras for any joint this frame — leave as NaN
            continue

        # Slice down to usable cameras only
        P_frame    = P_all[valid_cam_indices]           # (v, 3, 4)
        uv_frame   = frame_uv[valid_cam_indices]        # (v, num_joints, 2)
        conf_frame = frame_conf[valid_cam_indices]      # (v, num_joints)
        use_frame  = usable[valid_cam_indices]          # (v, num_joints) bool

        # Zero out confidence for invalid joints so they don't pull the DLT
        conf_frame = np.where(use_frame, conf_frame, 0.0)

        # Batched DLT: one SVD call returns all 17 joints at once
        pts_3d = batched_dlt_triangulate(P_frame, uv_frame, conf_frame)
        # pts_3d: (num_joints, 3)

        # Only store joints where at least min_cameras cameras were valid
        joint_cam_counts = use_frame.sum(axis=0)  # (num_joints,)
        sufficient       = joint_cam_counts >= min_cameras

        final_3d_timeline[frame_idx, sufficient] = pts_3d[sufficient]

        if do_optimization:
            # polish_3d_point can be updated to accept batched input when implemented
            for j in np.where(sufficient)[0]:
                final_3d_timeline[frame_idx, j] = polish_3d_point(
                    final_3d_timeline[frame_idx, j], P_frame, uv_frame[:, j, :]
                )

        if frame_idx % 300 == 0:
            print(f"  Frame {frame_idx}/{num_frames} — "
                  f"{sufficient.sum()}/{num_joints} joints triangulated")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 — INTERPOLATION (vectorized: all 3 axes in one interp1d call)
    # ─────────────────────────────────────────────────────────────────────────
    print("Running cubic spline interpolation to fix occlusions...")
    for joint_idx in range(num_joints):
        final_3d_timeline[:, joint_idx, :] = interpolate_missing_data(
            final_3d_timeline[:, joint_idx, :]
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4 — SAVE
    # interpolated flag marks joints that were NaN before phase 3 ran
    # ─────────────────────────────────────────────────────────────────────────
    print("Saving final 3D skeleton to JSON...")

    # Build an interpolated flag array: True where the DLT produced no result
    was_nan = np.isnan(final_3d_timeline[:, :, 0])  # will be False after interp fills them

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
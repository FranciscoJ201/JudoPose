import json
import numpy as np
import cv2
import os
from collections import defaultdict


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

# How many pixels of epipolar tolerance to allow when matching across cameras.
# Increase if your calibration is slightly noisy.
EPIPOLAR_THRESHOLD = 25.0

# Re-verify the cross-camera ID mapping every N frames.
# Lower = more robust to ID swaps, higher = faster.
REVERIFY_INTERVAL = 30

# Minimum HSV histogram correlation to trust a color match as a tiebreaker.
COLOR_MATCH_THRESHOLD = 0.6

# YOLOv8 keypoint indices for torso estimation
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12


# ─────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────

def get_torso_center(keypoints_2d):
    """
    Estimates the 2D torso center from YOLO keypoints.
    Uses the average of shoulders and hips, falling back to whatever is visible.

    Arguments:
    - keypoints_2d: List of [x, y, conf] for all 17 YOLO keypoints.

    Returns:
    - (u, v) float tuple, or None if no valid keypoints found.
    """
    torso_indices = [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP]
    valid_pts = []

    for idx in torso_indices:
        kp = keypoints_2d[idx]
        if kp[0] is not None and kp[1] is not None and float(kp[2]) > 0.3:
            valid_pts.append((float(kp[0]), float(kp[1])))

    if not valid_pts:
        return None

    u = np.mean([p[0] for p in valid_pts])
    v = np.mean([p[1] for p in valid_pts])
    return (u, v)


def compute_fundamental_matrix(P1, P2):
    """
    Derives the Fundamental Matrix F from two 3x4 projection matrices.
    F encodes the epipolar constraint: a point in cam1 maps to a LINE in cam2.

    Arguments:
    - P1, P2: 3x4 numpy projection matrices.

    Returns:
    - 3x3 numpy Fundamental Matrix.
    """
    # Extract camera centers from P = K[R|t] → center = -R^T t
    def camera_center(P):
        _, _, Vt = np.linalg.svd(P)
        C = Vt[-1]
        return C[:3] / C[3]

    C1 = camera_center(P1)

    # Epipole in image 2: the projection of camera 1's center into camera 2
    e2_h = P2 @ np.append(C1, 1.0)
    e2 = e2_h[:2] / e2_h[2]

    # Skew-symmetric matrix of e2
    e2x = np.array([
        [    0, -e2[2],  e2[1]],
        [ e2[2],     0, -e2[0]],
        [-e2[1],  e2[0],     0]
    ])

    # F = [e2]x @ P2 @ pinv(P1)
    F = e2x @ P2 @ np.linalg.pinv(P1)
    return F


def epipolar_distance(F, pt1, pt2):
    """
    Measures how well two 2D points satisfy the epipolar constraint.
    A low score means the points are geometrically consistent across cameras.

    Arguments:
    - F: 3x3 Fundamental Matrix.
    - pt1: (u, v) point in camera 1.
    - pt2: (u, v) point in camera 2.

    Returns:
    - Float: Symmetric epipolar distance in pixels.
    """
    p1_h = np.array([pt1[0], pt1[1], 1.0])
    p2_h = np.array([pt2[0], pt2[1], 1.0])

    # Epipolar line in image 2 corresponding to pt1
    l2 = F @ p1_h
    # Epipolar line in image 1 corresponding to pt2
    l1 = F.T @ p2_h

    # Symmetric distance: point-to-line distance in both images
    d2 = abs(p2_h @ l2) / np.sqrt(l2[0]**2 + l2[1]**2 + 1e-8)
    d1 = abs(p1_h @ l1) / np.sqrt(l1[0]**2 + l1[1]**2 + 1e-8)

    return (d1 + d2) / 2.0


# ─────────────────────────────────────────────
#  COLOR HELPER (tiebreaker)
# ─────────────────────────────────────────────

def extract_torso_color_histogram(frame_bgr, keypoints_2d, bins=16):
    """
    Extracts a compact HSV color histogram from the torso region.
    Used as a tiebreaker when epipolar geometry is ambiguous.

    Arguments:
    - frame_bgr: The raw BGR video frame as a numpy array.
    - keypoints_2d: YOLO keypoints for this person.
    - bins: Number of bins per HSV channel.

    Returns:
    - Normalized 1D numpy histogram, or None if no valid region found.
    """
    torso = get_torso_center(keypoints_2d)
    if torso is None:
        return None

    u, v = int(torso[0]), int(torso[1])
    h, w = frame_bgr.shape[:2]

    # Crop a 60x80 box around the torso center
    x1, x2 = max(0, u - 30), min(w, u + 30)
    y1, y2 = max(0, v - 40), min(h, v + 40)
    crop = frame_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def color_similarity(hist1, hist2):
    """
    Computes correlation between two HSV histograms. Range: -1 to 1, higher is more similar.
    """
    if hist1 is None or hist2 is None:
        return 0.0
    return cv2.compareHist(
        hist1.astype(np.float32),
        hist2.astype(np.float32),
        cv2.HISTCMP_CORREL
    )


# ─────────────────────────────────────────────
#  CROSS-CAMERA IDENTITY MATCHING
# ─────────────────────────────────────────────

def match_identities_across_cameras(frame_detections, calibration_dict, frames_bgr=None):
    """
    Assigns consistent global person IDs across all cameras for a single frame.

    Strategy:
    1. Camera A is the anchor — its detections define global IDs 0, 1, ...
    2. For each other camera, find the detection that best satisfies the
       epipolar constraint with each anchor person.
    3. If geometry is ambiguous (scores too close), use color as a tiebreaker.

    Arguments:
    - frame_detections: Dict mapping cam_name → list of detection dicts for this frame.
                        Each detection must have 'keypoints_2d'.
    - calibration_dict: Dict mapping cam_name → {'P': 3x4 numpy array, ...}
    - frames_bgr: Optional dict mapping cam_name → BGR numpy frame for color fallback.

    Returns:
    - Dict mapping cam_name → list of detection dicts, reordered so index = global person ID.
      Missing persons are represented as None.
    """
    cam_names = list(frame_detections.keys())
    if not cam_names:
        return {}

    # Camera A is the anchor
    anchor_cam = cam_names[0]
    anchor_detections = frame_detections[anchor_cam]
    num_people = len(anchor_detections)

    # Result structure: cam → [person_0_detection, person_1_detection, ...]
    aligned = {cam: [None] * num_people for cam in cam_names}
    aligned[anchor_cam] = anchor_detections[:num_people]

    # Precompute anchor torso centers
    anchor_centers = []
    for det in anchor_detections[:num_people]:
        center = get_torso_center(det.get('keypoints_2d', []))
        anchor_centers.append(center)

    # Match each non-anchor camera
    for cam_name in cam_names[1:]:
        other_detections = frame_detections[cam_name]
        if not other_detections:
            continue

        P_anchor = calibration_dict[anchor_cam]['P']
        P_other  = calibration_dict[cam_name]['P']
        F = compute_fundamental_matrix(P_anchor, P_other)

        # Build cost matrix: rows = anchor people, cols = other camera detections
        n_other = len(other_detections)
        cost_matrix = np.full((num_people, n_other), np.inf)

        for i, anchor_center in enumerate(anchor_centers):
            if anchor_center is None:
                continue
            for j, other_det in enumerate(other_detections):
                other_center = get_torso_center(other_det.get('keypoints_2d', []))
                if other_center is None:
                    continue
                cost_matrix[i, j] = epipolar_distance(F, anchor_center, other_center)

        # Greedy assignment: assign best geometric match first
        assigned_other = set()
        for i in range(num_people):
            best_j = -1
            best_cost = EPIPOLAR_THRESHOLD

            candidates = np.argsort(cost_matrix[i])
            for j in candidates:
                if j in assigned_other:
                    continue
                if cost_matrix[i, j] > EPIPOLAR_THRESHOLD:
                    break  # Already sorted, no point continuing

                # Check if this match is ambiguous (two candidates very close in score)
                ambiguous = False
                if len(candidates) > 1:
                    second_j = next((k for k in candidates if k != j and k not in assigned_other), None)
                    if second_j is not None:
                        score_gap = cost_matrix[i, second_j] - cost_matrix[i, j]
                        if score_gap < 5.0:  # pixels — very close scores
                            ambiguous = True

                # Use color as tiebreaker if ambiguous and frames are available
                if ambiguous and frames_bgr is not None:
                    anchor_frame = frames_bgr.get(anchor_cam)
                    other_frame  = frames_bgr.get(cam_name)

                    if anchor_frame is not None and other_frame is not None:
                        anchor_hist = extract_torso_color_histogram(
                            anchor_frame, anchor_detections[i].get('keypoints_2d', []))

                        # Compare anchor color against both candidates
                        best_color_j = j
                        best_color_score = -2.0
                        for cand_j in [j, second_j]:
                            if cand_j is None or cand_j in assigned_other:
                                continue
                            cand_hist = extract_torso_color_histogram(
                                other_frame, other_detections[cand_j].get('keypoints_2d', []))
                            sim = color_similarity(anchor_hist, cand_hist)
                            if sim > best_color_score:
                                best_color_score = sim
                                best_color_j = cand_j

                        if best_color_score >= COLOR_MATCH_THRESHOLD:
                            j = best_color_j

                best_j = j
                break

            if best_j >= 0:
                aligned[cam_name][i] = other_detections[best_j]
                assigned_other.add(best_j)

    return aligned


# ─────────────────────────────────────────────
#  MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────

def harmonize_person_ids(camera_json_paths, calibration_dict, output_dir,
                         video_paths=None, reverify_interval=REVERIFY_INTERVAL):
    """
    Loads per-camera YOLO JSONs, assigns globally consistent person IDs across
    all cameras and all frames, then saves new harmonized JSONs ready for triangulate.py.

    Arguments:
    - camera_json_paths: Dict mapping cam_name → path to YOLO output JSON.
                         Each JSON is a list of frame dicts, each with a 'detections' key.
    - calibration_dict:  Dict mapping cam_name → {'P': 3x4 numpy array, ...}
                         (same format as used in triangulate.py)
    - output_dir:        Directory to write the harmonized JSONs.
                         Output files are named: {cam_name}_harmonized.json
    - video_paths:       Optional dict mapping cam_name → video file path.
                         Provide this to enable color-based tiebreaking.
    - reverify_interval: Re-run cross-camera matching every N frames to catch
                         any ID swaps that BoTSORT missed.

    Returns:
    - Dict mapping cam_name → path to the harmonized output JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all YOLO JSONs
    print("Loading YOLO tracking JSONs...")
    cam_data = {}
    for cam_name, path in camera_json_paths.items():
        with open(path, 'r') as f:
            cam_data[cam_name] = json.load(f)
        print(f"  {cam_name}: {len(cam_data[cam_name])} frames loaded from {path}")

    num_frames = min(len(data) for data in cam_data.values())
    cam_names  = list(cam_data.keys())

    # Open video captures for color fallback if provided
    video_caps = {}
    if video_paths:
        for cam_name, vpath in video_paths.items():
            cap = cv2.VideoCapture(vpath)
            if cap.isOpened():
                video_caps[cam_name] = cap
            else:
                print(f"  Warning: Could not open video for {cam_name}, color fallback disabled.")

    # Output structure mirrors input: list of frame dicts
    harmonized = {cam: [] for cam in cam_names}

    # Track the last verified cross-camera mapping: anchor_person_idx → other_cam detections
    # This lets us skip expensive re-matching on most frames
    last_mapping_frame = -reverify_interval  # Force match on frame 0

    print(f"Harmonizing {num_frames} frames (re-verifying every {reverify_interval} frames)...")

    for frame_idx in range(num_frames):

        # Grab BGR frames for color fallback if video caps are open
        frames_bgr = {}
        for cam_name, cap in video_caps.items():
            ret, frame = cap.read()
            if ret:
                frames_bgr[cam_name] = frame

        # Collect detections for this frame across all cameras
        frame_detections = {}
        for cam_name in cam_names:
            frame_dict = cam_data[cam_name][frame_idx]
            frame_detections[cam_name] = frame_dict.get('detections', [])

        # Decide whether to re-run cross-camera matching this frame
        needs_reverify = (frame_idx - last_mapping_frame) >= reverify_interval

        if needs_reverify:
            aligned = match_identities_across_cameras(
                frame_detections, calibration_dict,
                frames_bgr if frames_bgr else None
            )
            last_mapping_frame = frame_idx
        else:
            # On non-reverify frames, rely on BoTSORT within each camera
            # and just reorder by the stable track IDs (no geometric re-matching needed)
            aligned = {cam: frame_detections[cam] for cam in cam_names}

        # Write harmonized frame to each camera's output
        for cam_name in cam_names:
            original_frame = cam_data[cam_name][frame_idx]
            harmonized_frame = {
                "frame_index": frame_idx,
                "detections": aligned.get(cam_name, [])
            }
            # Preserve any other metadata in the original frame dict
            for key, val in original_frame.items():
                if key not in harmonized_frame:
                    harmonized_frame[key] = val

            harmonized[cam_name].append(harmonized_frame)

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{num_frames}")

    # Release video captures
    for cap in video_caps.values():
        cap.release()

    # Save harmonized JSONs
    output_paths = {}
    print("Saving harmonized JSONs...")
    for cam_name in cam_names:
        out_path = os.path.join(output_dir, f"{cam_name}_harmonized.json")
        with open(out_path, 'w') as f:
            json.dump(harmonized[cam_name], f, indent=4)
        output_paths[cam_name] = out_path
        print(f"  Saved: {out_path}")

    print("Done! Feed these harmonized JSONs into triangulate.py.")
    return output_paths



   
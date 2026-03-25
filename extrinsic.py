import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import glob

# ─────────────────────────────────────────────
#  BOARD CONFIGURATION
#
#  80mm squares, 5x7 layout → physical size 400mm × 560mm
#  Board is held VERTICALLY with base flush on the mat floor.
#  This means:
#    - Bottom-left inner corner of board is at (placement_x, placement_y, 0.0)
#    - Corners increase in Z (upward) as row index increases
#    - Corners increase in X or Y depending on board facing direction
#
#  IMPORTANT: Measure your printed squares with a ruler before use.
#  A 1% printer scale error = ~4mm error at the top of the board.
# ─────────────────────────────────────────────

SQUARE_SIZE   = 0.040          # 80mm squares in metres (change 4 to 8 when update and then the 3 to a 6)
SQUARES_X     = 5              # columns
SQUARES_Y     = 7              # rows
MARKER_SIZE   = 0.030          # ArUco marker inside each square
ARUCO_DICT    = aruco.DICT_4X4_50

# Reprojection error threshold — refuse to save if mean pixel error exceeds this.
# 1.5px is a reasonable bar for 300+ point correspondences.
MAX_REPROJ_ERROR = 1.5

# The 5 known world positions where the board base will be placed.
# These are derived from the mat corner clicks in Phase 1 + the mat centre.
# Keys match the position labels used in the capture folder structure.
# Values are (X, Y) in metres in your world coordinate system.
# Z is always 0.0 at the base — the board extends upward from the floor.
BOARD_POSITIONS = {
    'top_left':     None,   # Filled in from Phase 1 mat corner clicks
    'top_right':    None,
    'bottom_right': None,
    'bottom_left':  None,
    'centre':       (0.0, 0.0),   # Mat centre is always the world origin
}

# At each position the board is held at 3 rotations:
#   0°  — facing directly toward the RealSense (or primary camera)
#  45°  — rotated left
# -45°  — rotated right
# This ensures every camera in the rig sees the board face-on at least once.
# The rotation affects which world axis the board columns run along.
# 'facing_x' means columns run along X axis, rows run up Z.
# 'facing_y' means columns run along Y axis, rows run up Z.
# '45deg'    means columns run along the diagonal — approximated as equal X+Y.
BOARD_ROTATIONS = ['facing_x', 'facing_y', 'diagonal']


# ─────────────────────────────────────────────
#  GLOBAL STATE FOR MOUSE CALLBACK
# ─────────────────────────────────────────────

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    """Captures mouse clicks for mat corner definition (Phase 1)."""
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((float(x), float(y)))
            img_copy = param.copy()

            for i, p in enumerate(clicked_points):
                cv2.circle(img_copy, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
                cv2.putText(img_copy, str(i + 1),
                            (int(p[0]) + 10, int(p[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if i > 0:
                    cv2.line(img_copy,
                             (int(clicked_points[i-1][0]), int(clicked_points[i-1][1])),
                             (int(p[0]), int(p[1])), (0, 255, 255), 2)

            if len(clicked_points) == 4:
                cv2.line(img_copy,
                         (int(clicked_points[3][0]), int(clicked_points[3][1])),
                         (int(clicked_points[0][0]), int(clicked_points[0][1])),
                         (0, 255, 255), 2)
                print("  4 corners recorded. Press any key to continue...")

            cv2.imshow("PHASE 1 — Click the 4 mat corners", img_copy)


# ─────────────────────────────────────────────
#  PHASE 1 — MAT CORNER CLICKING
#  Defines the world coordinate system.
#  Returns the 4 mat corner world coords and their pixel coords.
# ─────────────────────────────────────────────

def phase1_click_mat_corners(undistorted_img, mat_size_meters):
    """
    Shows the undistorted frame and asks the user to click the 4 mat corners.
    These clicks anchor the world coordinate system:
        - Mat centre = (0, 0, 0)
        - X runs along the long axis (left→right)
        - Y runs along the short axis (bottom→top)
        - Z points straight up

    Click order: Top-Left, Top-Right, Bottom-Right, Bottom-Left

    Arguments:
    - undistorted_img:  BGR numpy array, already undistorted.
    - mat_size_meters:  Full side length of the square combat area in metres.

    Returns:
    - image_points_2d:  (4, 2) float32 array of clicked pixel coords.
    - world_points_3d:  (4, 3) float32 array of known world coords.
    - corner_world_xy:  Dict mapping position name → (X, Y) world coords.
                        Used to fill in BOARD_POSITIONS for Phase 2.
    """
    global clicked_points
    clicked_points = []

    half = mat_size_meters / 2.0

    # World coords of the 4 corners — Z=0 (floor plane)
    world_pts = np.array([
        [-half,  half, 0.0],   # Top-Left
        [ half,  half, 0.0],   # Top-Right
        [ half, -half, 0.0],   # Bottom-Right
        [-half, -half, 0.0],   # Bottom-Left
    ], dtype=np.float32)

    corner_world_xy = {
        'top_left':     (-half,  half),
        'top_right':    ( half,  half),
        'bottom_right': ( half, -half),
        'bottom_left':  (-half, -half),
    }

    print("\n--- PHASE 1: World Coordinate Anchoring ---")
    print("Click the 4 corners of the Judo combat area in this exact order:")
    print("  1. Top-Left")
    print("  2. Top-Right")
    print("  3. Bottom-Right")
    print("  4. Bottom-Left")
    print("These clicks define where (0,0,0) is and which way X and Y point.")
    print("Press any key after clicking all 4 corners.\n")

    cv2.imshow("PHASE 1 — Click the 4 mat corners", undistorted_img)
    cv2.setMouseCallback("PHASE 1 — Click the 4 mat corners",
                         mouse_callback, undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicked_points) != 4:
        raise RuntimeError(f"Expected 4 clicks, got {len(clicked_points)}. Aborting.")

    image_pts = np.array(clicked_points, dtype=np.float32)
    return image_pts, world_pts, corner_world_xy


# ─────────────────────────────────────────────
#  BOARD WORLD COORDINATES BUILDER
#
#  Given the board's base position (bx, by) in world coords and its
#  facing direction, compute the 3D world coordinate of every inner
#  ChArUco corner on the board.
#
#  The board has (SQUARES_X - 1) × (SQUARES_Y - 1) inner corners.
#  Corner (col, row) in board-local coordinates maps to world coords as:
#
#    facing_x:   world = (bx + col*sq,  by,              row*sq)
#    facing_y:   world = (bx,           by + col*sq,     row*sq)
#    diagonal:   world = (bx + col*sq*cos45, by + col*sq*sin45, row*sq)
#
#  Row 0 is at the bottom (Z=0), row increases upward.
#  This matches how cv2.aruco.CharucoBoard lays out corner IDs.
# ─────────────────────────────────────────────

def build_board_world_coords(base_xy, facing, square_size=SQUARE_SIZE,
                              squares_x=SQUARES_X, squares_y=SQUARES_Y):
    """
    Computes 3D world coordinates for all inner ChArUco corners on a vertically
    held board at a known base position.

    Arguments:
    - base_xy:     (X, Y) world position of the board's bottom-left corner in metres.
    - facing:      One of 'facing_x', 'facing_y', 'diagonal'.
    - square_size: Physical size of one square in metres (default 0.080m).
    - squares_x:   Number of squares horizontally on the board.
    - squares_y:   Number of squares vertically on the board.

    Returns:
    - world_coords: numpy array of shape (num_inner_corners, 3) — one row per
                    inner corner, ordered to match CharucoBoard corner IDs.
    """
    bx, by = base_xy
    inner_cols = squares_x - 1
    inner_rows = squares_y - 1

    cos45 = np.cos(np.radians(45))
    sin45 = np.sin(np.radians(45))

    coords = []
    # CharucoBoard orders corners row-major from top-left of the board image.
    # For a vertical board: row 0 = top = highest Z, row increases downward = lower Z.
    for row in range(inner_rows):
        z = (inner_rows - 1 - row) * square_size   # row 0 → highest Z
        for col in range(inner_cols):
            if facing == 'facing_x':
                wx = bx + col * square_size
                wy = by
            elif facing == 'facing_y':
                wx = bx
                wy = by + col * square_size
            else:  # diagonal
                wx = bx + col * square_size * cos45
                wy = by + col * square_size * sin45
            coords.append([wx, wy, z])

    return np.array(coords, dtype=np.float32)


# ─────────────────────────────────────────────
#  PHASE 2 — CHARUCO BOARD DETECTION
#  Loads all board images for one camera, detects corners, maps them
#  to world coordinates using the known placement positions.
# ─────────────────────────────────────────────

def phase2_detect_charuco_boards(board_image_folder, corner_world_xy):
    """
    Detects ChArUco corners in all board images and returns matched
    (image_point, world_point) pairs for use in the final solvePnP call.

    Folder structure expected:
        board_image_folder/
            top_left_facing_x/     ← position_rotation subfolders
            top_left_facing_y/
            top_left_diagonal/
            top_right_facing_x/
            ... (5 positions × 3 rotations = 15 subfolders)
            centre_facing_x/
            ...

    Arguments:
    - board_image_folder: Root folder containing position_rotation subfolders.
    - corner_world_xy:    Dict from Phase 1 mapping position → (X, Y) world coords.
                          Centre is always (0.0, 0.0).

    Returns:
    - all_image_pts: (N, 1, 2) float32 array — detected pixel coordinates.
    - all_world_pts: (N, 1, 3) float32 array — corresponding world coordinates.
    - num_images_used: int — number of images where detection succeeded.
    """
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    board      = aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_SIZE, MARKER_SIZE, dictionary)
    detector   = aruco.CharucoDetector(board)

    all_image_pts = []
    all_world_pts = []
    num_images_used = 0

    # Build the full position → (X, Y) mapping including centre
    position_xy = {**corner_world_xy, 'centre': (0.0, 0.0)}

    for position_name, base_xy in position_xy.items():
        if base_xy is None:
            print(f"  Warning: No world coords for position '{position_name}' — skipping.")
            continue

        for facing in BOARD_ROTATIONS:
            subfolder = os.path.join(board_image_folder,
                                     f"{position_name}_{facing}")
            images = (glob.glob(os.path.join(subfolder, '*.jpg')) +
                      glob.glob(os.path.join(subfolder, '*.png')))

            if not images:
                print(f"  Warning: No images found in {subfolder}")
                continue

            # Precompute world coords for this position+facing combination
            board_world = build_board_world_coords(base_xy, facing)

            for img_path in images:
                img  = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

                if charuco_corners is None or charuco_ids is None:
                    continue
                if len(charuco_corners) < 6:
                    # Too few corners — unreliable detection, skip this image
                    continue

                # charuco_ids indexes into the board's inner corner array.
                # board_world is already ordered to match CharucoBoard corner IDs.
                ids_flat = charuco_ids.flatten()

                # Guard against IDs outside the board's corner count
                max_id = (SQUARES_X - 1) * (SQUARES_Y - 1) - 1
                valid_mask = ids_flat <= max_id
                if not np.any(valid_mask):
                    continue

                valid_corners = charuco_corners[valid_mask]
                valid_ids     = ids_flat[valid_mask]
                valid_world   = board_world[valid_ids]   # (n, 3)

                # Reshape to OpenCV's expected (n, 1, 2) and (n, 1, 3) formats
                all_image_pts.append(
                    valid_corners.reshape(-1, 1, 2).astype(np.float32))
                all_world_pts.append(
                    valid_world.reshape(-1, 1, 3).astype(np.float32))

                num_images_used += 1

    if not all_image_pts:
        raise RuntimeError(
            "No ChArUco corners detected in any board image. "
            "Check folder structure and image quality.")

    combined_img   = np.concatenate(all_image_pts, axis=0)
    combined_world = np.concatenate(all_world_pts, axis=0)

    print(f"  Phase 2: {num_images_used} images used, "
          f"{len(combined_img)} total point correspondences collected.")

    return combined_img, combined_world, num_images_used


# ─────────────────────────────────────────────
#  REPROJECTION ERROR CHECK
# ─────────────────────────────────────────────

def compute_reprojection_error(world_pts, image_pts, rvec, tvec, K, dist):
    """
    Projects world points through the solved camera pose and measures
    mean pixel distance from the detected image points.

    A value below 1.5px for 300+ points indicates a reliable calibration.
    Above 2.0px suggests a problem — bad images, wrong board size, or
    incorrect board placement world coordinates.

    Arguments:
    - world_pts: (N, 1, 3) float32 world coordinates.
    - image_pts: (N, 1, 2) float32 detected pixel coordinates.
    - rvec, tvec: Rotation and translation from solvePnP.
    - K, dist:   Camera matrix and distortion (zero dist if undistorted).

    Returns:
    - mean_error: float — mean reprojection error in pixels.
    - max_error:  float — worst single point error in pixels.
    """
    projected, _ = cv2.projectPoints(world_pts, rvec, tvec, K, dist)
    projected     = projected.reshape(-1, 2)
    detected      = image_pts.reshape(-1, 2)

    errors     = np.linalg.norm(projected - detected, axis=1)
    mean_error = float(np.mean(errors))
    max_error  = float(np.max(errors))

    return mean_error, max_error


# ─────────────────────────────────────────────
#  UNDISTORTION HELPER
# ─────────────────────────────────────────────

def undistort_image_and_get_new_K(img, K, D, camera_model):
    """
    Undistorts an image using the appropriate model for this camera type.
    Returns the undistorted image and the new (optimal) camera matrix.

    After undistortion, all downstream work (clicking, ChArUco detection,
    solvePnP) uses new_K with zero distortion coefficients.

    Arguments:
    - img:          BGR numpy array.
    - K:            Original camera matrix from intrinsic calibration.
    - D:            Distortion coefficients from intrinsic calibration.
    - camera_model: 'standard', 'fisheye_rational', or 'fisheye'.

    Returns:
    - undistorted:  BGR numpy array.
    - new_K:        3x3 optimal camera matrix for the undistorted image.
    """
    h, w = img.shape[:2]

    if camera_model == 'fisheye':
        # Legacy GoPro fisheye model
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.5)
        undistorted = cv2.fisheye.undistortImage(img, K, D, Knew=new_K)
    else:
        # 'standard' (RealSense) and 'fisheye_rational' (GoPro rational model)
        # both use the standard undistort pipeline
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, K, D, None, new_K)

    return undistorted, new_K


# ─────────────────────────────────────────────
#  MAIN CALIBRATION FUNCTION
# ─────────────────────────────────────────────

def calibrate_extrinsics_combined(
        anchor_image_path,
        board_image_folder,
        intrinsic_json,
        output_json,
        mat_size_meters=8.0):
    """
    Combined Option 1 + Option 3 extrinsic calibration.

    Phase 1 — Click 4 mat corners on a clean frame to anchor the world
              coordinate system. This defines what (0,0,0) means and
              which direction X and Y point. These 4 points are included
              in the final solvePnP call.

    Phase 2 — Detect ChArUco corners from images of the vertical board
              held at 5 known positions (4 mat corners + centre) in 3
              rotations each. These provide ~300+ point correspondences
              spanning X, Y, and Z — breaking the coplanar degeneracy
              that makes flat-board calibration unreliable for wide-angle
              cameras at low mounting angles.

    Phase 3 — Combine all points, run solvePnP with SOLVEPNP_ITERATIVE,
              compute reprojection error, refuse to save if error exceeds
              MAX_REPROJ_ERROR (1.5px).

    Folder structure for board_image_folder:
        board_images/
            top_left_facing_x/       ← 2-3 JPGs of board at top-left, facing X
            top_left_facing_y/
            top_left_diagonal/
            top_right_facing_x/
            top_right_facing_y/
            top_right_diagonal/
            bottom_right_facing_x/
            bottom_right_facing_y/
            bottom_right_diagonal/
            bottom_left_facing_x/
            bottom_left_facing_y/
            bottom_left_diagonal/
            centre_facing_x/
            centre_facing_y/
            centre_diagonal/

    Arguments:
    - anchor_image_path:  Path to a clean frame with no athletes — used for
                          Phase 1 mat corner clicking.
    - board_image_folder: Root folder containing the 15 position_rotation
                          subfolders of vertical board images.
    - intrinsic_json:     Path to this camera's intrinsic calibration JSON.
    - output_json:        Where to save the resulting extrinsic JSON.
    - mat_size_meters:    Full side length of the square combat area (default 8.0m).

    Returns:
    - True on success, False on failure (bad reprojection error or solvePnP failure).
    """

    # ── Load intrinsics ──────────────────────────────────────────────────────
    if not os.path.exists(intrinsic_json):
        print(f"Error: Intrinsic file not found: {intrinsic_json}")
        return False

    with open(intrinsic_json, 'r') as f:
        intr = json.load(f)

    K            = np.array(intr['camera_matrix'],           dtype=np.float64)
    D            = np.array(intr['distortion_coefficients'], dtype=np.float64)
    camera_model = intr.get('camera_model', 'standard')

    # ── Load and undistort the anchor frame ─────────────────────────────────
    anchor_img = cv2.imread(anchor_image_path)
    if anchor_img is None:
        print(f"Error: Could not load anchor image: {anchor_image_path}")
        return False

    print(f"Undistorting anchor image ({camera_model} model)...")
    undistorted_anchor, new_K = undistort_image_and_get_new_K(
        anchor_img, K, D, camera_model)

    # From here all work uses new_K and zero distortion
    zero_dist = np.zeros((4, 1), dtype=np.float64)

    # ── Phase 1 — Click mat corners ──────────────────────────────────────────
    mat_img_pts, mat_world_pts, corner_world_xy = phase1_click_mat_corners(
        undistorted_anchor, mat_size_meters)

    print(f"\nPhase 1 complete — 4 mat corners recorded.")
    for name, xy in corner_world_xy.items():
        print(f"  {name}: world ({xy[0]:.2f}, {xy[1]:.2f}, 0.00) m")

    # ── Phase 2 — ChArUco board detection ────────────────────────────────────
    print(f"\n--- PHASE 2: Vertical ChArUco Board Detection ---")
    print(f"Looking for board images in: {board_image_folder}")
    print(f"Board spec: {SQUARES_X}×{SQUARES_Y} squares, {SQUARE_SIZE*1000:.0f}mm each")
    print(f"Expected subfolders: 5 positions × 3 rotations = 15 subfolders\n")

    # Board images also need to be undistorted before detection.
    # We load raw images, undistort them, then run ChArUco detection.
    # The undistorted images are processed in-memory — not saved to disk.
    board_img_pts_raw, board_world_pts, num_board_images = \
        phase2_detect_charuco_boards_undistorted(
            board_image_folder, corner_world_xy, K, D, camera_model, new_K)

    print(f"Phase 2 complete — {num_board_images} images processed.")

    # ── Phase 3 — Combine all points and solve ───────────────────────────────
    print(f"\n--- PHASE 3: Combined solvePnP ---")

    # Stack mat corner points with board points
    # mat_img_pts: (4, 2) → reshape to (4, 1, 2)
    # mat_world_pts: (4, 3) → reshape to (4, 1, 3)
    mat_img_pts_r   = mat_img_pts.reshape(-1, 1, 2).astype(np.float32)
    mat_world_pts_r = mat_world_pts.reshape(-1, 1, 3).astype(np.float32)

    all_img_pts   = np.concatenate([mat_img_pts_r,   board_img_pts_raw],  axis=0)
    all_world_pts = np.concatenate([mat_world_pts_r, board_world_pts], axis=0)

    print(f"Total point correspondences: {len(all_img_pts)} "
          f"(4 mat corners + {len(board_img_pts_raw)} board corners)")
    print(f"Running solvePnP (ITERATIVE)...")

    success, rvec, tvec = cv2.solvePnP(
        all_world_pts.reshape(-1, 1, 3),
        all_img_pts.reshape(-1, 1, 2),
        new_K, zero_dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("ERROR: solvePnP failed. Check that board world coordinates "
              "match actual physical placement and images are undistorted.")
        return False

    # ── Reprojection error check ─────────────────────────────────────────────
    mean_err, max_err = compute_reprojection_error(
        all_world_pts, all_img_pts, rvec, tvec, new_K, zero_dist)

    print(f"Reprojection error — mean: {mean_err:.3f}px  max: {max_err:.3f}px")

    if mean_err > MAX_REPROJ_ERROR:
        print(f"\nERROR: Mean reprojection error ({mean_err:.3f}px) exceeds "
              f"threshold ({MAX_REPROJ_ERROR}px).")
        print("Possible causes:")
        print("  - Board square size set incorrectly (measure with a ruler)")
        print("  - Board placement position does not match world coordinates")
        print("  - Board was not truly vertical during capture")
        print("  - Mat corner clicks were inaccurate")
        print("Calibration NOT saved. Fix the issue and re-run.")
        return False

    # ── Build and save output ────────────────────────────────────────────────
    R, _ = cv2.Rodrigues(rvec)
    RT   = np.hstack((R, tvec))
    P    = new_K @ RT

    # Camera position in world coords: C = -R^T @ t
    cam_pos_world = (-R.T @ tvec).flatten()

    extrinsic_data = {
        "rotation_matrix":       R.tolist(),
        "translation_vector":    tvec.tolist(),
        "projection_matrix":     P.tolist(),
        "optimal_camera_matrix": new_K.tolist(),
        "camera_position_world": cam_pos_world.tolist(),  # (X, Y, Z) of camera
        "reprojection_error_px": round(mean_err, 4),
        "num_point_correspondences": int(len(all_img_pts)),
        "calibration_method":    "option1+3_mat_corners_vertical_charuco"
    }

    with open(output_json, 'w') as f:
        json.dump(extrinsic_data, f, indent=4)

    print(f"\nSUCCESS!")
    print(f"  Camera world position : X={cam_pos_world[0]:.3f}m  "
          f"Y={cam_pos_world[1]:.3f}m  Z={cam_pos_world[2]:.3f}m (height)")
    print(f"  Reprojection error    : {mean_err:.3f}px (threshold {MAX_REPROJ_ERROR}px)")
    print(f"  Point correspondences : {len(all_img_pts)}")
    print(f"  Saved to              : {output_json}")
    return True


# ─────────────────────────────────────────────
#  PHASE 2 WITH IN-MEMORY UNDISTORTION
#  Board images are undistorted before ChArUco detection — same as
#  the anchor frame — so all detected pixel coords are in the
#  new_K coordinate system with zero distortion.
# ─────────────────────────────────────────────

def phase2_detect_charuco_boards_undistorted(
        board_image_folder, corner_world_xy, K, D, camera_model, new_K):
    """
    Same as phase2_detect_charuco_boards but undistorts each board image
    before detection. This is essential — ChArUco corner pixel coords must
    be in the same undistorted coordinate system as the mat corner clicks.

    Arguments:
    - board_image_folder: Root folder with position_rotation subfolders.
    - corner_world_xy:    Dict from Phase 1.
    - K, D:               Original intrinsics for undistortion.
    - camera_model:       Camera distortion model string.
    - new_K:              Optimal camera matrix after undistortion.

    Returns:
    - all_image_pts: (N, 1, 2) float32
    - all_world_pts: (N, 1, 3) float32
    - num_images_used: int
    """
    zero_dist  = np.zeros((4, 1), dtype=np.float64)
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    board      = aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_SIZE, MARKER_SIZE, dictionary)
    detector   = aruco.CharucoDetector(board)

    all_image_pts = []
    all_world_pts = []
    num_images_used = 0

    position_xy = {**corner_world_xy, 'centre': (0.0, 0.0)}

    for position_name, base_xy in position_xy.items():
        if base_xy is None:
            continue

        for facing in BOARD_ROTATIONS:
            subfolder = os.path.join(board_image_folder,
                                     f"{position_name}_{facing}")
            images = (glob.glob(os.path.join(subfolder, '*.jpg')) +
                      glob.glob(os.path.join(subfolder, '*.png')))

            if not images:
                print(f"  Warning: No images in {subfolder}")
                continue

            board_world = build_board_world_coords(base_xy, facing)

            for img_path in images:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Undistort before detection — critical for accuracy
                undist, _ = undistort_image_and_get_new_K(
                    img, K, D, camera_model)
                gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

                charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

                if charuco_corners is None or charuco_ids is None:
                    continue
                if len(charuco_corners) < 6:
                    continue

                ids_flat = charuco_ids.flatten()
                max_id   = (SQUARES_X - 1) * (SQUARES_Y - 1) - 1
                valid    = ids_flat <= max_id

                if not np.any(valid):
                    continue

                valid_corners = charuco_corners[valid]
                valid_world   = board_world[ids_flat[valid]]

                all_image_pts.append(
                    valid_corners.reshape(-1, 1, 2).astype(np.float32))
                all_world_pts.append(
                    valid_world.reshape(-1, 1, 3).astype(np.float32))
                num_images_used += 1

    if not all_image_pts:
        raise RuntimeError(
            "No ChArUco corners detected in any board image. "
            "Check subfolder names match: {position}_{facing} "
            "(e.g. top_left_facing_x)")

    return (np.concatenate(all_image_pts, axis=0),
            np.concatenate(all_world_pts, axis=0),
            num_images_used)


# ─────────────────────────────────────────────
#  CAPTURE GUIDE — printed at runtime to remind
#  the operator exactly what to capture and where
# ─────────────────────────────────────────────

def print_capture_guide(mat_size_meters=8.0):
    """
    Prints a step-by-step physical capture guide for the operator.
    Run this before going to the gym so you know exactly what to capture.
    """
    half = mat_size_meters / 2.0
    print("\n" + "="*60)
    print("  EXTRINSIC CALIBRATION CAPTURE GUIDE")
    print("="*60)
    print(f"\nBoard spec: 5×7 squares, 80mm each → 400mm × 560mm physical size")
    print(f"IMPORTANT: Measure your printed squares before use.")
    print(f"           Update SQUARE_SIZE in this file if they differ from 80mm.")
    print(f"\nCoordinate system:")
    print(f"  Origin (0,0,0) = mat centre")
    print(f"  X axis         = long axis of mat (left→right)")
    print(f"  Y axis         = short axis of mat (bottom→top)")
    print(f"  Z axis         = straight up from floor")
    print(f"\nMat corner world coordinates at mat_size={mat_size_meters}m:")
    print(f"  Top-Left     : ({-half:.1f},  {half:.1f}, 0.0)")
    print(f"  Top-Right    : ( {half:.1f},  {half:.1f}, 0.0)")
    print(f"  Bottom-Right : ( {half:.1f}, {-half:.1f}, 0.0)")
    print(f"  Bottom-Left  : ({-half:.1f}, {-half:.1f}, 0.0)")
    print(f"\nPHASE 1 — Anchor frame (no board, no athletes)")
    print(f"  - Take one clean frame from each camera showing the full mat")
    print(f"  - You will click the 4 mat corners in the UI")
    print(f"\nPHASE 2 — Vertical board capture (all cameras simultaneously)")
    print(f"  Board holder: wear plain dark clothing, hold board truly vertical")
    print(f"  Use a spirit level taped to the board frame (±2° tolerance)")
    print(f"  Board base must be flush on the mat floor (Z=0)")
    print(f"\n  5 positions × 3 rotations = 15 subfolders per camera")
    print(f"  Capture 2-3 images per subfolder (hold still for 2-3 seconds)")
    print(f"\n  Positions:")
    print(f"    top_left      — board base at mat Top-Left corner")
    print(f"    top_right     — board base at mat Top-Right corner")
    print(f"    bottom_right  — board base at mat Bottom-Right corner")
    print(f"    bottom_left   — board base at mat Bottom-Left corner")
    print(f"    centre        — board base at mat centre (0,0,0)")
    print(f"\n  Rotations at each position:")
    print(f"    facing_x   — board face pointing along X axis (toward long edge)")
    print(f"    facing_y   — board face pointing along Y axis (toward short edge)")
    print(f"    diagonal   — board face pointing at 45° diagonal")
    print(f"\n  Subfolder naming: {{position}}_{{rotation}}")
    print(f"  Example: top_left_facing_x/  top_left_facing_y/  top_left_diagonal/")
    print(f"\n  Total images per camera: ~30-45 (2-3 per subfolder × 15 subfolders)")
    print(f"\nPHASE 3 — Solve (automatic)")
    print(f"  Target reprojection error: < {MAX_REPROJ_ERROR}px")
    print(f"  If error is too high, check board square size measurement first.")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Print the physical capture guide before going to the gym
    print_capture_guide(mat_size_meters=8.0)

    # Example usage — run once per camera after capturing board images:
    #
    # calibrate_extrinsics_combined(
    #     anchor_image_path  = 'calib_frames/cam_a_anchor.jpg',
    #     board_image_folder = 'board_images/cam_a/',
    #     intrinsic_json     = 'intrinsic_cam_a.json',
    #     output_json        = 'extrinsic_cam_a.json',
    #     mat_size_meters    = 8.0
    # )
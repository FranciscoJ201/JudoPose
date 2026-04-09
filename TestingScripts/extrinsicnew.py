import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import glob

# ─────────────────────────────────────────────
#  BOARD CONFIGURATION
#
#  Board lies FLAT on the mat floor (Z = 0 plane).
#  The board's own local coordinate system is used by OpenCV to solve
#  board-to-camera pose per image. That pose is then composed with the
#  board's known world position (clicked by the user in Phase 1) to
#  accumulate world↔pixel correspondences for the final solvePnP.
#
#  IMPORTANT: Measure your printed squares with a ruler before use.
#  A 1% printer scale error → ~4mm positional error across the board.
# ─────────────────────────────────────────────

SQUARE_SIZE      = 0.040           # metres (80mm squares)
SQUARES_X        = 5               # columns
SQUARES_Y        = 7               # rows
MARKER_SIZE      = 0.030           # ArUco marker size in metres
ARUCO_DICT       = aruco.DICT_4X4_50

# Refuse to save calibration if mean reprojection error exceeds this.
MAX_REPROJ_ERROR = 1.5             # pixels

# Minimum ChArUco corners required per image to use it.
MIN_CORNERS_PER_IMAGE = 6


# ─────────────────────────────────────────────
#  GLOBAL STATE FOR MOUSE CALLBACK
# ─────────────────────────────────────────────

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    """Records up to 4 left-click pixel coordinates on the displayed image."""
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((float(x), float(y)))
        img_copy = param.copy()

        for i, p in enumerate(clicked_points):
            cv2.circle(img_copy, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
            cv2.putText(img_copy, str(i + 1),
                        (int(p[0]) + 10, int(p[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if i > 0:
                cv2.line(img_copy,
                         (int(clicked_points[i - 1][0]), int(clicked_points[i - 1][1])),
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
#  Defines the world coordinate system origin and axes.
# ─────────────────────────────────────────────

def phase1_click_mat_corners(undistorted_img, mat_size_meters):
    """
    Shows the undistorted anchor frame. The user clicks the 4 mat corners
    in order: Top-Left → Top-Right → Bottom-Right → Bottom-Left.

    World coordinate system:
        Origin (0,0,0) = mat centre
        X  — long axis, left → right
        Y  — short axis, bottom → top
        Z  — straight up from floor

    Arguments:
        undistorted_img  : BGR numpy array, already undistorted.
        mat_size_meters  : Full side length of the square combat area (metres).

    Returns:
        image_pts   : (4, 2) float32 — clicked pixel coordinates.
        world_pts   : (4, 3) float32 — corresponding world coordinates (Z=0).
    """
    global clicked_points
    clicked_points = []

    half = mat_size_meters / 2.0

    world_pts = np.array([
        [-half,  half, 0.0],   # Top-Left
        [ half,  half, 0.0],   # Top-Right
        [ half, -half, 0.0],   # Bottom-Right
        [-half, -half, 0.0],   # Bottom-Left
    ], dtype=np.float32)

    print("\n--- PHASE 1: World Coordinate Anchoring ---")
    print("Click the 4 corners of the Judo mat in this exact order:")
    print("  1. Top-Left")
    print("  2. Top-Right")
    print("  3. Bottom-Right")
    print("  4. Bottom-Left")
    print("Press any key after all 4 clicks.\n")

    cv2.imshow("PHASE 1 — Click the 4 mat corners", undistorted_img)
    cv2.setMouseCallback("PHASE 1 — Click the 4 mat corners",
                         mouse_callback, undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicked_points) != 4:
        raise RuntimeError(
            f"Expected 4 clicks, got {len(clicked_points)}. Aborting.")

    image_pts = np.array(clicked_points, dtype=np.float32)
    return image_pts, world_pts


# ─────────────────────────────────────────────
#  UNDISTORTION HELPER
# ─────────────────────────────────────────────

def undistort_image(img, K, D, camera_model, new_K):
    """
    Undistorts a single image using the pre-computed optimal camera matrix.
    All downstream pixel coordinates are in the new_K coordinate system
    with zero distortion.

    Arguments:
        img          : BGR numpy array.
        K            : Original camera matrix.
        D            : Distortion coefficients.
        camera_model : 'standard', 'fisheye_rational', or 'fisheye'.
        new_K        : Pre-computed optimal camera matrix (from get_new_K).

    Returns:
        undistorted  : BGR numpy array.
    """
    if camera_model == 'fisheye':
        return cv2.fisheye.undistortImage(img, K, D, Knew=new_K)
    else:
        return cv2.undistort(img, K, D, None, new_K)


def get_new_K(img_shape, K, D, camera_model):
    """
    Computes the optimal camera matrix for the undistorted image space.
    Call once and reuse for every image from the same camera.

    Arguments:
        img_shape    : (height, width) of the images.
        K            : Original camera matrix.
        D            : Distortion coefficients.
        camera_model : 'standard', 'fisheye_rational', or 'fisheye'.

    Returns:
        new_K        : (3, 3) float64 optimal camera matrix.
    """
    h, w = img_shape
    if camera_model == 'fisheye':
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.5)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    return new_K


# ─────────────────────────────────────────────
#  PHASE 2 — FLAT BOARD DETECTION
#
#  For each board image:
#    1. Undistort the image.
#    2. Detect ChArUco corners → get corner IDs and pixel coords.
#    3. Use estimatePoseCharucoBoard to get board-to-camera pose (rvec, tvec).
#       This is OpenCV solving "where is this board relative to the camera"
#       using the board's own internal metric coordinate system (Z=0 plane).
#    4. The user clicks the board origin position on the anchor frame to
#       give us the board's world XY position. We compose the board-to-camera
#       pose with the board's world transform to get world↔pixel pairs.
#
#  No subfolders, no facing directions, no manual world coord construction.
#  Drop any flat board images into one folder and this handles the rest.
# ─────────────────────────────────────────────

def phase2_detect_flat_boards(board_image_folder, K, D, camera_model, new_K):
    """
    Detects ChArUco corners in all flat board images and returns matched
    (image_point, world_point) pairs for use in the final solvePnP.

    The board can be placed anywhere on the mat in any flat orientation.
    OpenCV's estimatePoseCharucoBoard recovers the board's pose relative to
    the camera; we then express each corner in world space using that pose
    and the camera's current best estimate (bootstrapped from Phase 1 clicks).

    Since we don't know the camera pose yet at this stage, we accumulate
    board-local corner coordinates and their detected pixel positions. The
    board's own Z=0 plane is the mat floor, so board-local coords ARE world
    coords as long as the board is flat. The board origin in world space is
    determined by the board's position on the mat — which we handle by
    solving iteratively: first solve from Phase 1 corners alone, then use
    that pose to transform board corners into world space.

    Arguments:
        board_image_folder : Folder containing flat board images (JPG/PNG).
                             No subfolders required — all images are processed.
        K, D               : Original camera intrinsics.
        camera_model       : 'standard', 'fisheye_rational', or 'fisheye'.
        new_K              : Pre-computed optimal camera matrix.

    Returns:
        all_image_pts  : (N, 1, 2) float32 — detected pixel coordinates.
        all_board_pts  : (N, 1, 3) float32 — board-local 3D coords (Z=0 plane).
        num_images_used: int
    """
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    board      = aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_SIZE, MARKER_SIZE, dictionary)
    detector   = aruco.CharucoDetector(board)

    # Board-local 3D coordinates of every inner corner.
    # CharucoBoard places corner (col, row) at (col*sq, row*sq, 0) in board space.
    board_obj_pts = board.getChessboardCorners()   # (N_corners, 3) float32

    images = (glob.glob(os.path.join(board_image_folder, '*.jpg')) +
              glob.glob(os.path.join(board_image_folder, '*.png')) +
              glob.glob(os.path.join(board_image_folder, '*.JPG')) +
              glob.glob(os.path.join(board_image_folder, '*.PNG')))

    if not images:
        raise RuntimeError(
            f"No board images found in: {board_image_folder}\n"
            "Expected JPG or PNG files directly in this folder.")

    all_image_pts  = []
    all_board_pts  = []
    num_images_used = 0
    zero_dist      = np.zeros((4, 1), dtype=np.float64)

    for img_path in sorted(images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Could not read {img_path} — skipping.")
            continue

        undist = undistort_image(img, K, D, camera_model, new_K)
        gray   = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_corners is None or charuco_ids is None:
            print(f"  Warning: No ChArUco corners detected in {os.path.basename(img_path)}")
            continue
        if len(charuco_corners) < MIN_CORNERS_PER_IMAGE:
            print(f"  Warning: Only {len(charuco_corners)} corners in "
                  f"{os.path.basename(img_path)} — need {MIN_CORNERS_PER_IMAGE}, skipping.")
            continue

        ids_flat     = charuco_ids.flatten()
        max_valid_id = len(board_obj_pts) - 1
        valid_mask   = ids_flat <= max_valid_id

        if not np.any(valid_mask):
            continue

        valid_pixel_pts = charuco_corners[valid_mask]           # (n, 1, 2)
        valid_board_pts = board_obj_pts[ids_flat[valid_mask]]   # (n, 3)

        all_image_pts.append(
            valid_pixel_pts.reshape(-1, 1, 2).astype(np.float32))
        all_board_pts.append(
            valid_board_pts.reshape(-1, 1, 3).astype(np.float32))

        num_images_used += 1
        print(f"  {os.path.basename(img_path):40s}  "
              f"{len(valid_pixel_pts):3d} corners accepted")

    if not all_image_pts:
        raise RuntimeError(
            "No usable board images found. Check image quality and board visibility.")

    print(f"\n  Phase 2: {num_images_used}/{len(images)} images used.")

    return (np.concatenate(all_image_pts, axis=0),
            np.concatenate(all_board_pts, axis=0),
            num_images_used)


# ─────────────────────────────────────────────
#  REPROJECTION ERROR
# ─────────────────────────────────────────────

def compute_reprojection_error(world_pts, image_pts, rvec, tvec, K, dist):
    """
    Projects world points through the solved camera pose and measures
    mean pixel distance from detected image points.

    Returns:
        mean_error : float — mean reprojection error in pixels.
        max_error  : float — worst single-point error in pixels.
    """
    projected, _ = cv2.projectPoints(world_pts, rvec, tvec, K, dist)
    projected     = projected.reshape(-1, 2)
    detected      = image_pts.reshape(-1, 2)
    errors        = np.linalg.norm(projected - detected, axis=1)
    return float(np.mean(errors)), float(np.max(errors))


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
    Extrinsic calibration using mat corner clicks + flat ChArUco board images.

    Phase 1 — Click 4 mat corners on a clean anchor frame to anchor the
              world coordinate system. Origin = mat centre, Z = up.

    Phase 2 — Detect ChArUco corners from flat board images placed anywhere
              on the mat. The board can be in any position and orientation
              as long as it lies flat (Z=0). No subfolders or naming
              conventions required — just drop images into one folder.
              OpenCV's CharucoBoard provides the board-local 3D coordinates
              for each detected corner directly. Since the board is flat on
              the mat (Z=0), board-local XY coords align with the mat floor
              plane. We use an initial camera pose estimate (from the 4 mat
              corner clicks) to rotate the board-local points into world space.

    Phase 3 — Combine all point correspondences, run solvePnP, check
              reprojection error, save if below MAX_REPROJ_ERROR.

    Arguments:
        anchor_image_path  : Clean frame showing the full mat (no athletes).
        board_image_folder : Folder of flat board images (all in one folder).
        intrinsic_json     : Path to this camera's intrinsic calibration JSON.
        output_json        : Output path for extrinsic JSON.
        mat_size_meters    : Full side length of the square combat area (metres).

    Returns:
        True on success, False on failure.

    Output JSON keys:
        rotation_matrix          — (3, 3) world-to-camera rotation
        translation_vector       — (3, 1) world-to-camera translation
        projection_matrix        — (3, 4) P = new_K @ [R | t]
        optimal_camera_matrix    — (3, 3) new_K (undistorted space)
        camera_position_world    — [X, Y, Z] camera centre in world coords
        reprojection_error_px    — mean reprojection error
        num_point_correspondences
        calibration_method
    """

    # ── Load intrinsics ──────────────────────────────────────────────────────
    if not os.path.exists(intrinsic_json):
        print(f"Error: Intrinsic file not found: {intrinsic_json}")
        return False

    with open(intrinsic_json, 'r') as f:
        intr = json.load(f)

    if 'camera_matrix' in intr:
        K             = np.array(intr['camera_matrix'], dtype=np.float64)
        D             = np.array(intr['distortion_coefficients'], dtype=np.float64)
        camera_model  = intr.get('camera_model', 'standard')
    else:
        fx = intr['fx']; fy = intr['fy']
        cx = intr['cx']; cy = intr['cy']
        K  = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]], dtype=np.float64)
        D  = np.array(intr['distortion_coeffs'], dtype=np.float64)
        camera_model = 'standard'

    # ── Load anchor frame and compute new_K once ─────────────────────────────
    anchor_img = cv2.imread(anchor_image_path)
    if anchor_img is None:
        print(f"Error: Could not load anchor image: {anchor_image_path}")
        return False

    img_shape = anchor_img.shape[:2]   # (h, w)
    new_K     = get_new_K(img_shape, K, D, camera_model)
    zero_dist = np.zeros((4, 1), dtype=np.float64)

    print(f"Undistorting anchor image ({camera_model} model)...")
    undistorted_anchor = undistort_image(anchor_img, K, D, camera_model, new_K)

    # ── Phase 1 — Click mat corners ──────────────────────────────────────────
    mat_img_pts, mat_world_pts = phase1_click_mat_corners(
        undistorted_anchor, mat_size_meters)

    print(f"\nPhase 1 complete — 4 mat corners recorded.")

    # ── Bootstrap camera pose from mat corners alone ─────────────────────────
    # This initial estimate is used to transform board-local coords into world
    # space for the board images collected in Phase 2.
    success_init, rvec_init, tvec_init = cv2.solvePnP(
        mat_world_pts.reshape(-1, 1, 3),
        mat_img_pts.reshape(-1, 1, 2),
        new_K, zero_dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success_init:
        print("Error: Initial solvePnP from mat corners failed. "
              "Check that all 4 corners were clicked accurately.")
        return False

    R_init, _ = cv2.Rodrigues(rvec_init)

    # ── Phase 2 — Flat board detection ───────────────────────────────────────
    print(f"\n--- PHASE 2: Flat ChArUco Board Detection ---")
    print(f"Looking for board images in: {board_image_folder}")
    print(f"Board spec: {SQUARES_X}×{SQUARES_Y} squares, "
          f"{SQUARE_SIZE * 1000:.0f}mm each (flat on mat floor)\n")

    board_img_pts, board_local_pts, num_board_images = phase2_detect_flat_boards(
        board_image_folder, K, D, camera_model, new_K)

    # Transform board-local points into world space using the initial camera pose.
    # Board-local coords have Z=0 (flat on mat floor). The board can be placed
    # anywhere — its position and yaw in world space are recovered by projecting
    # the board corners through the initial camera pose and finding where they
    # land on the Z=0 world plane.
    #
    # Concretely: for each board corner with board-local position p_board,
    # we back-project through the camera to get its world position:
    #   p_cam   = R_init @ p_world + t_init   (from Phase 1 solve)
    #   p_world = R_init^T @ (p_cam - t_init)
    #
    # But we don't know p_cam directly. Instead we use the detected pixel
    # coords and the initial pose to cast a ray and intersect with Z=0.
    # The simpler and more robust approach: use estimatePoseCharucoBoard
    # (board→camera) and compose with camera→world (inverse of Phase 1 pose)
    # to get board corners directly in world space.

    board_world_pts = _transform_board_pts_to_world(
        board_img_pts, board_local_pts, new_K, zero_dist, R_init, tvec_init)

    print(f"Phase 2 complete — {num_board_images} images, "
          f"{len(board_world_pts)} point correspondences.")

    # ── Phase 3 — Combine all points and solve ───────────────────────────────
    print(f"\n--- PHASE 3: Combined solvePnP ---")

    mat_img_pts_r   = mat_img_pts.reshape(-1, 1, 2).astype(np.float32)
    mat_world_pts_r = mat_world_pts.reshape(-1, 1, 3).astype(np.float32)

    all_img_pts   = np.concatenate([mat_img_pts_r,   board_img_pts],   axis=0)
    all_world_pts = np.concatenate([mat_world_pts_r, board_world_pts], axis=0)

    print(f"Total correspondences: {len(all_img_pts)} "
          f"(4 mat corners + {len(board_img_pts)} board corners)")
    print("Running solvePnP (ITERATIVE)...")

    success, rvec, tvec = cv2.solvePnP(
        all_world_pts.reshape(-1, 1, 3),
        all_img_pts.reshape(-1, 1, 2),
        new_K, zero_dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("Error: solvePnP failed on combined point set.")
        return False

    # ── Reprojection error ───────────────────────────────────────────────────
    mean_err, max_err = compute_reprojection_error(
        all_world_pts, all_img_pts, rvec, tvec, new_K, zero_dist)

    print(f"Reprojection error — mean: {mean_err:.3f}px  max: {max_err:.3f}px")

    if mean_err > MAX_REPROJ_ERROR:
        print(f"\nError: Mean reprojection error ({mean_err:.3f}px) exceeds "
              f"threshold ({MAX_REPROJ_ERROR}px).")
        print("Possible causes:")
        print("  - Board square size set incorrectly (measure with a ruler)")
        print("  - Board was not flat on the mat during capture")
        print("  - Mat corner clicks were inaccurate")
        print("Calibration NOT saved.")
        return False

    # ── Build and save output ────────────────────────────────────────────────
    R, _  = cv2.Rodrigues(rvec)
    RT    = np.hstack((R, tvec))
    P     = new_K @ RT
    cam_pos_world = (-R.T @ tvec).flatten()

    extrinsic_data = {
        "rotation_matrix":            R.tolist(),
        "translation_vector":         tvec.tolist(),
        "projection_matrix":          P.tolist(),
        "optimal_camera_matrix":      new_K.tolist(),
        "camera_position_world":      cam_pos_world.tolist(),
        "reprojection_error_px":      round(mean_err, 4),
        "num_point_correspondences":  int(len(all_img_pts)),
        "calibration_method":         "mat_corners_flat_charuco",
    }

    with open(output_json, 'w') as f:
        json.dump(extrinsic_data, f, indent=4)

    print(f"\nSUCCESS!")
    print(f"  Camera world position : X={cam_pos_world[0]:.3f}m  "
          f"Y={cam_pos_world[1]:.3f}m  Z={cam_pos_world[2]:.3f}m (height)")
    print(f"  Reprojection error    : {mean_err:.3f}px  "
          f"(threshold {MAX_REPROJ_ERROR}px)")
    print(f"  Point correspondences : {len(all_img_pts)}")
    print(f"  Saved to              : {output_json}")
    return True


# ─────────────────────────────────────────────
#  BOARD-LOCAL → WORLD TRANSFORM
#
#  Given detected board pixel coords and board-local 3D coords (from
#  CharucoBoard.getChessboardCorners), recover each corner's world position.
#
#  Strategy: for each image, use solvePnP to get board→camera pose, then
#  compose with camera→world (inverse of the Phase 1 pose) to express
#  board corners in world space. This is exact and doesn't require knowing
#  where the board was placed beforehand.
# ─────────────────────────────────────────────

def _transform_board_pts_to_world(board_img_pts, board_local_pts,
                                   new_K, zero_dist, R_cam, tvec_cam):
    """
    Transforms board-local 3D coordinates into world coordinates by composing
    the board→camera pose with the camera→world transform.

    For each detected corner:
        p_cam   = R_board_cam @ p_board + t_board_cam   (board→camera)
        p_world = R_cam^T @ (p_cam - tvec_cam)          (camera→world)

    Arguments:
        board_img_pts   : (N, 1, 2) float32 — all detected pixel coords.
        board_local_pts : (N, 1, 3) float32 — corresponding board-local coords.
        new_K           : (3, 3) optimal camera matrix.
        zero_dist       : Zero distortion (images already undistorted).
        R_cam           : (3, 3) world-to-camera rotation from Phase 1 solve.
        tvec_cam        : (3, 1) world-to-camera translation from Phase 1 solve.

    Returns:
        world_pts : (N, 1, 3) float32 — board corner positions in world space.
    """
    # We process all points together — each "image" contributed a contiguous
    # block of corners. We need to recover the per-image board pose, so we
    # re-group by image. Since we concatenated across images in Phase 2, we
    # work on the full set but use a per-chunk solvePnP approach by iterating
    # over the image contributions tracked via unique local positions.
    #
    # Simpler: since board_local_pts has Z=0 (flat board), we can directly
    # use the per-image solvePnP result to recover board orientation + position
    # relative to camera, then unproject to world.

    # Flatten for batch processing
    img_pts_flat   = board_img_pts.reshape(-1, 2)
    local_pts_flat = board_local_pts.reshape(-1, 3)

    # Use solvePnP on the full combined set to get an average board→camera
    # transform. This is a simplification — ideally we'd do per-image. But
    # since Z=0 for all board points and the combined set spans the full mat,
    # the world coords recovered this way are accurate.
    #
    # Per-image approach: detect unique "sessions" by proximity in local_pts.
    # Actually the cleanest solution: re-run detection per image and compose.
    # We do that below by exploiting the fact that board_local_pts repeated
    # corner IDs identify the same physical corners across images.

    # Find unique board corners by local position (rounded to mm)
    # and for each unique corner, take the mean of its detected pixel positions.
    # This gives one robust pixel location per board corner per placement.
    # Then solve once per unique set of board positions (i.e., per image placement).
    #
    # Simplest correct approach for a flat board: since all board points have
    # Z=0 in board space, and the board is flat on the mat (Z=0 in world space),
    # board-local X and Y map directly to world X and Y *up to a rigid 2D transform*
    # (translation + rotation in the Z=0 plane). We recover that transform via
    # the camera pose, then apply it.

    # camera→world transform (inverse of Phase 1 solve)
    R_world = R_cam.T                                # world←camera rotation
    t_world = -R_cam.T @ tvec_cam                   # world←camera translation

    world_pts_list = []

    # Group corners by image by re-processing — we need per-image board pose.
    # Since we don't have per-image boundaries here, we solve on the full set.
    # For a flat board (Z=0), solvePnP gives a valid average if the board
    # didn't move much, but for multiple placements this won't be right.
    #
    # Correct approach: phase2 should return per-image data. We restructure
    # _transform_board_pts_to_world to accept per-image lists and process each.

    # NOTE: This function is called with the full concatenated arrays.
    # The transform is applied point-by-point using the camera→world matrix,
    # but we still need per-image board→camera pose to unproject correctly.
    # See _collect_world_pts_per_image for the correct implementation.

    raise NotImplementedError(
        "_transform_board_pts_to_world should not be called directly. "
        "Use _collect_world_pts_per_image instead.")


def _collect_world_pts_per_image(board_image_folder, K, D, camera_model,
                                  new_K, R_cam, tvec_cam):
    """
    Per-image version of board-to-world transform. For each board image:
      1. Detect ChArUco corners → pixel coords + board-local 3D coords.
      2. solvePnP → board→camera pose (rvec_bc, tvec_bc).
      3. Compose: p_world = R_cam^T @ (R_bc @ p_board + t_bc) - R_cam^T @ t_cam

    This is the function actually used in the pipeline. Phase 2 detection
    and world-space transform are combined here to avoid storing redundant
    intermediate arrays.

    Arguments:
        board_image_folder : Folder of flat board images.
        K, D               : Original camera intrinsics.
        camera_model       : Distortion model string.
        new_K              : Pre-computed optimal camera matrix.
        R_cam              : (3, 3) world-to-camera rotation (from Phase 1).
        tvec_cam           : (3, 1) world-to-camera translation (from Phase 1).

    Returns:
        all_image_pts  : (N, 1, 2) float32 — pixel coords in undistorted space.
        all_world_pts  : (N, 1, 3) float32 — corresponding world coords.
        num_images_used: int
    """
    dictionary    = aruco.getPredefinedDictionary(ARUCO_DICT)
    board         = aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_SIZE, MARKER_SIZE, dictionary)
    detector      = aruco.CharucoDetector(board)
    board_obj_pts = board.getChessboardCorners()   # (N_corners, 3) float32

    images = sorted(
        glob.glob(os.path.join(board_image_folder, '*.jpg')) +
        glob.glob(os.path.join(board_image_folder, '*.png')) +
        glob.glob(os.path.join(board_image_folder, '*.JPG')) +
        glob.glob(os.path.join(board_image_folder, '*.PNG'))
    )

    if not images:
        raise RuntimeError(
            f"No board images found in: {board_image_folder}")

    zero_dist      = np.zeros((4, 1), dtype=np.float64)
    R_world        = R_cam.T
    t_world        = -R_cam.T @ tvec_cam      # camera origin in world space

    all_image_pts  = []
    all_world_pts  = []
    num_images_used = 0
    skipped         = 0

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Could not read {img_path} — skipping.")
            skipped += 1
            continue

        undist = undistort_image(img, K, D, camera_model, new_K)
        gray   = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_corners is None or charuco_ids is None:
            print(f"  Warning: No detections in {os.path.basename(img_path)}")
            skipped += 1
            continue
        if len(charuco_corners) < MIN_CORNERS_PER_IMAGE:
            print(f"  Warning: {len(charuco_corners)} corners in "
                  f"{os.path.basename(img_path)} — need {MIN_CORNERS_PER_IMAGE}, skipping.")
            skipped += 1
            continue

        ids_flat     = charuco_ids.flatten()
        max_valid_id = len(board_obj_pts) - 1
        valid_mask   = ids_flat <= max_valid_id

        if not np.any(valid_mask):
            skipped += 1
            continue

        valid_pixel  = charuco_corners[valid_mask]              # (n, 1, 2)
        valid_local  = board_obj_pts[ids_flat[valid_mask]]      # (n, 3)

        # Solve board→camera pose using board-local coords (Z=0 plane)
        ok, rvec_bc, tvec_bc = cv2.solvePnP(
            valid_local.reshape(-1, 1, 3),
            valid_pixel.reshape(-1, 1, 2),
            new_K, zero_dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok:
            print(f"  Warning: Board pose solve failed for "
                  f"{os.path.basename(img_path)} — skipping.")
            skipped += 1
            continue

        R_bc, _ = cv2.Rodrigues(rvec_bc)

        # Transform board-local corners to world space:
        #   p_cam   = R_bc @ p_board + t_bc
        #   p_world = R_world @ p_cam + t_world
        #           = R_world @ (R_bc @ p_board + t_bc) + t_world
        p_board  = valid_local.T                        # (3, n)
        p_cam    = R_bc @ p_board + tvec_bc             # (3, n)
        p_world  = R_world @ p_cam + t_world            # (3, n)

        world_pts = p_world.T.reshape(-1, 1, 3).astype(np.float32)

        all_image_pts.append(valid_pixel.reshape(-1, 1, 2).astype(np.float32))
        all_world_pts.append(world_pts)
        num_images_used += 1

        print(f"  {os.path.basename(img_path):40s}  "
              f"{len(valid_pixel):3d} corners")

    if not all_image_pts:
        raise RuntimeError(
            "No usable board images. Check image quality and board visibility.")

    print(f"\n  Phase 2: {num_images_used} used, {skipped} skipped.")

    return (np.concatenate(all_image_pts, axis=0),
            np.concatenate(all_world_pts, axis=0),
            num_images_used)


# ─────────────────────────────────────────────
#  REVISED MAIN — wires up _collect_world_pts_per_image
# ─────────────────────────────────────────────

def calibrate_extrinsics_combined(
        anchor_image_path,
        board_image_folder,
        intrinsic_json,
        output_json,
        mat_size_meters=8.0):
    """
    Extrinsic calibration: mat corner clicks + flat ChArUco board images.

    Capture protocol:
        1. Take one clean anchor frame of the full mat (no athletes, no board).
        2. Place the board flat on the mat in ~8-10 different positions,
           covering the mat area. Orientation doesn't matter — any flat
           placement works. Take one photo per position from each camera.
        3. Drop all board images into one folder per camera. No subfolders.
        4. Run this function once per camera.

    Arguments:
        anchor_image_path  : Clean frame showing the full mat.
        board_image_folder : Flat folder of board images (JPG/PNG).
        intrinsic_json     : Camera intrinsic calibration JSON.
        output_json        : Output path for extrinsic JSON.
        mat_size_meters    : Full side length of the combat area (metres).

    Returns:
        True on success, False on failure.
    """

    # ── Load intrinsics ──────────────────────────────────────────────────────
    if not os.path.exists(intrinsic_json):
        print(f"Error: Intrinsic file not found: {intrinsic_json}")
        return False

    with open(intrinsic_json, 'r') as f:
        intr = json.load(f)

    if 'camera_matrix' in intr:
        K            = np.array(intr['camera_matrix'], dtype=np.float64)
        D            = np.array(intr['distortion_coefficients'], dtype=np.float64)
        camera_model = intr.get('camera_model', 'standard')
    else:
        fx = intr['fx']; fy = intr['fy']
        cx = intr['cx']; cy = intr['cy']
        K  = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]], dtype=np.float64)
        D  = np.array(intr['distortion_coeffs'], dtype=np.float64)
        camera_model = 'standard'

    # ── Load anchor frame, compute new_K once ───────────────────────────────
    anchor_img = cv2.imread(anchor_image_path)
    if anchor_img is None:
        print(f"Error: Could not load anchor image: {anchor_image_path}")
        return False

    new_K     = get_new_K(anchor_img.shape[:2], K, D, camera_model)
    zero_dist = np.zeros((4, 1), dtype=np.float64)

    print(f"Camera model : {camera_model}")
    print(f"Anchor image : {anchor_image_path}")
    undistorted_anchor = undistort_image(anchor_img, K, D, camera_model, new_K)

    # ── Phase 1 — Click mat corners ──────────────────────────────────────────
    mat_img_pts, mat_world_pts = phase1_click_mat_corners(
        undistorted_anchor, mat_size_meters)
    print(f"Phase 1 complete — 4 mat corners recorded.")

    # ── Bootstrap initial camera pose from mat corners ────────────────────────
    ok_init, rvec_init, tvec_init = cv2.solvePnP(
        mat_world_pts.reshape(-1, 1, 3),
        mat_img_pts.reshape(-1, 1, 2),
        new_K, zero_dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok_init:
        print("Error: Could not solve initial pose from mat corners. "
              "Check that clicks are accurate and in the correct order.")
        return False

    R_init, _ = cv2.Rodrigues(rvec_init)

    init_err, _ = compute_reprojection_error(
        mat_world_pts.reshape(-1, 1, 3),
        mat_img_pts.reshape(-1, 1, 2),
        rvec_init, tvec_init, new_K, zero_dist)
    print(f"Initial pose reprojection error (4 mat corners): {init_err:.3f}px")

    # ── Phase 2 — Flat board detection + world transform ─────────────────────
    print(f"\n--- PHASE 2: Flat ChArUco Board Detection ---")
    print(f"Folder : {board_image_folder}")
    print(f"Board  : {SQUARES_X}×{SQUARES_Y} squares, "
          f"{SQUARE_SIZE * 1000:.0f}mm each, flat on mat\n")

    board_img_pts, board_world_pts, num_board_images = \
        _collect_world_pts_per_image(
            board_image_folder, K, D, camera_model,
            new_K, R_init, tvec_init)

    # ── Phase 3 — Combined solvePnP ───────────────────────────────────────────
    print(f"\n--- PHASE 3: Combined solvePnP ---")

    all_img_pts   = np.concatenate([
        mat_img_pts.reshape(-1, 1, 2).astype(np.float32),
        board_img_pts
    ], axis=0)
    all_world_pts = np.concatenate([
        mat_world_pts.reshape(-1, 1, 3).astype(np.float32),
        board_world_pts
    ], axis=0)

    print(f"Total correspondences : {len(all_img_pts)} "
          f"(4 mat corners + {len(board_img_pts)} board corners)")
    print("Running solvePnP (ITERATIVE)...")

    success, rvec, tvec = cv2.solvePnP(
        all_world_pts.reshape(-1, 1, 3),
        all_img_pts.reshape(-1, 1, 2),
        new_K, zero_dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("Error: Final solvePnP failed.")
        return False

    mean_err, max_err = compute_reprojection_error(
        all_world_pts, all_img_pts, rvec, tvec, new_K, zero_dist)

    print(f"Reprojection error — mean: {mean_err:.3f}px  max: {max_err:.3f}px")

    if mean_err > MAX_REPROJ_ERROR:
        print(f"\nError: Reprojection error ({mean_err:.3f}px) exceeds "
              f"threshold ({MAX_REPROJ_ERROR}px).")
        print("Possible causes:")
        print("  - Board square size incorrect (measure with a ruler and update SQUARE_SIZE)")
        print("  - Board was not lying flat on the mat")
        print("  - Mat corner clicks were inaccurate (re-run and click precisely)")
        print("  - Too few board positions — aim for 8+ spread across the mat")
        print("Calibration NOT saved.")
        return False

    # ── Save output ──────────────────────────────────────────────────────────
    R, _          = cv2.Rodrigues(rvec)
    RT            = np.hstack((R, tvec))
    P             = new_K @ RT
    cam_pos_world = (-R.T @ tvec).flatten()

    extrinsic_data = {
        "rotation_matrix":           R.tolist(),
        "translation_vector":        tvec.tolist(),
        "projection_matrix":         P.tolist(),
        "optimal_camera_matrix":     new_K.tolist(),
        "camera_position_world":     cam_pos_world.tolist(),
        "reprojection_error_px":     round(mean_err, 4),
        "num_point_correspondences": int(len(all_img_pts)),
        "calibration_method":        "mat_corners_flat_charuco",
    }

    with open(output_json, 'w') as f:
        json.dump(extrinsic_data, f, indent=4)

    print(f"\nSUCCESS!")
    print(f"  Camera world position : X={cam_pos_world[0]:.3f}m  "
          f"Y={cam_pos_world[1]:.3f}m  Z={cam_pos_world[2]:.3f}m (height)")
    print(f"  Reprojection error    : {mean_err:.3f}px  "
          f"(threshold {MAX_REPROJ_ERROR}px)")
    print(f"  Point correspondences : {len(all_img_pts)}")
    print(f"  Saved to              : {output_json}")
    return True


# ─────────────────────────────────────────────
#  CAPTURE GUIDE
# ─────────────────────────────────────────────

def print_capture_guide(mat_size_meters=8.0):
    """Prints the physical capture protocol for the operator."""
    half = mat_size_meters / 2.0
    print("\n" + "=" * 60)
    print("  EXTRINSIC CALIBRATION CAPTURE GUIDE")
    print("=" * 60)
    print(f"\nBoard spec: {SQUARES_X}×{SQUARES_Y} squares, "
          f"{SQUARE_SIZE*1000:.0f}mm each")
    print(f"IMPORTANT: Measure your printed squares with a ruler.")
    print(f"           Update SQUARE_SIZE if they differ from "
          f"{SQUARE_SIZE*1000:.0f}mm.")
    print(f"\nCoordinate system:")
    print(f"  Origin (0,0,0) = mat centre")
    print(f"  X  — long axis, left → right")
    print(f"  Y  — short axis, bottom → top")
    print(f"  Z  — straight up from floor")
    print(f"\nMat corner world coordinates (mat_size={mat_size_meters}m):")
    print(f"  Top-Left     : ({-half:.1f},  {half:.1f}, 0.0)")
    print(f"  Top-Right    : ( {half:.1f},  {half:.1f}, 0.0)")
    print(f"  Bottom-Right : ( {half:.1f}, {-half:.1f}, 0.0)")
    print(f"  Bottom-Left  : ({-half:.1f}, {-half:.1f}, 0.0)")
    print(f"\nPHASE 1 — Anchor frame")
    print(f"  - One clean frame per camera, full mat visible, no athletes")
    print(f"  - You will click the 4 mat corners in the UI")
    print(f"\nPHASE 2 — Flat board images")
    print(f"  - Place the board flat on the mat (any orientation is fine)")
    print(f"  - Aim for 8-12 positions spread across the mat area")
    print(f"  - One photo per position per camera")
    print(f"  - All images go in one flat folder — no subfolders needed")
    print(f"  - Board holder: step out of frame after placing the board")
    print(f"\nPHASE 3 — Solve (automatic)")
    print(f"  Target reprojection error: < {MAX_REPROJ_ERROR}px")
    print(f"  If error is too high, check SQUARE_SIZE first.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_capture_guide(mat_size_meters=8.0)

    # Example usage:
    #
    # calibrate_extrinsics_combined(
    #     anchor_image_path  = 'calib_frames/cam_a_anchor.jpg',
    #     board_image_folder = 'board_images/cam_a/',
    #     intrinsic_json     = 'intrinsic_cam_a.json',
    #     output_json        = 'extrinsic_cam_a.json',
    #     mat_size_meters    = 8.0
    # )
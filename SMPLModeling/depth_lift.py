"""
depth_lift.py
=============
Back-projects YOLO keypoints from 2-D image space into 3-D camera space using
the aligned depth frames saved by RealSenseRecord60.py.

YOLO v8/11 pose keypoints (17 COCO keypoints, 0-indexed):
  0  nose           1  left_eye       2  right_eye
  3  left_ear       4  right_ear      5  left_shoulder
  6  right_shoulder 7  left_elbow     8  right_elbow
  9  left_wrist    10  right_wrist   11  left_hip
 12  right_hip     13  left_knee     14  right_knee
 15  left_ankle    16  right_ankle

SMPL body joints we need (24 joints, subset mapped from COCO):
The SMPLFitter will handle the incomplete mapping — we just pass all 17 kps.
"""

import json
import os
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Minimum YOLO keypoint visibility/confidence to trust
KP_CONF_THRESHOLD = 0.3

# Depth sanity range in metres  (discard noisy/missing depth)
MIN_DEPTH_M = 0.2
MAX_DEPTH_M = 8.0

# Kernel half-size for median depth sampling around a keypoint pixel
DEPTH_PATCH_HALF = 3   # 7×7 window


# ─────────────────────────────────────────────────────────────────────────────
#  DEPTH LIFTER
# ─────────────────────────────────────────────────────────────────────────────

class DepthLifter:
    """
    Reads camera intrinsics + aligned depth .npy frames and lifts each
    YOLO keypoint from (u, v) → (X, Y, Z) in camera space.
    """

    def __init__(self, intrinsics_path: str, depth_dir: str):
        self.depth_dir = Path(depth_dir)
        self._load_intrinsics(intrinsics_path)

    # ── Intrinsics ────────────────────────────────────────────────────────────

    def _load_intrinsics(self, path: str):
        with open(path, "r") as f:
            intr = json.load(f)

        self.fx = float(intr["fx"])
        self.fy = float(intr["fy"])
        self.cx = float(intr["cx"])
        self.cy = float(intr["cy"])
        self.W  = int(intr["width"])
        self.H  = int(intr["height"])

        print(f"[DepthLifter] Intrinsics loaded: fx={self.fx:.2f} fy={self.fy:.2f} "
              f"cx={self.cx:.2f} cy={self.cy:.2f}  res={self.W}×{self.H}")

    # ── Depth reader ──────────────────────────────────────────────────────────

    def _load_depth(self, frame_index: int) -> np.ndarray | None:
        """
        Load depth_XXXXXX.npy (uint16, millimetres as saved by RealSenseRecord60).
        Returns float32 array in METRES, or None if file missing.
        """
        npy_path = self.depth_dir / f"depth_{frame_index:06d}.npy"
        if not npy_path.exists():
            return None
        depth_mm = np.load(str(npy_path)).astype(np.float32)
        return depth_mm / 1000.0   # mm → metres

    # ── Pixel → 3-D ──────────────────────────────────────────────────────────

    def _sample_depth(self, depth_m: np.ndarray, u: int, v: int) -> float:
        """
        Robust depth at pixel (u, v) using median of a small neighbourhood,
        ignoring zero/invalid pixels.
        """
        h, w = depth_m.shape
        r = DEPTH_PATCH_HALF
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)
        patch   = depth_m[v0:v1, u0:u1]
        valid   = patch[(patch > MIN_DEPTH_M) & (patch < MAX_DEPTH_M)]
        if valid.size == 0:
            return 0.0
        return float(np.median(valid))

    def _backproject(self, u: float, v: float, z: float) -> np.ndarray:
        """Pinhole back-projection: (u,v,z) → (X,Y,Z) camera coords."""
        X = (u - self.cx) * z / self.fx
        Y = (v - self.cy) * z / self.fy
        return np.array([X, Y, z], dtype=np.float32)

    # ── Per-frame lifting ─────────────────────────────────────────────────────

    def lift_frame(self, detection: dict) -> dict | None:
        """
        Lift one detection dict (from poseestimation.py JSON) to 3-D.

        Returns
        -------
        dict with keys:
            frame_index  : int
            track_id     : int
            kps_3d       : np.ndarray  shape (17, 3)   — (X,Y,Z) metres, NaN if invalid
            kps_conf     : np.ndarray  shape (17,)      — YOLO visibility score
            bbox_xywh    : list
            det_conf     : float
        """
        frame_idx = detection["frame_index"]
        depth_m   = self._load_depth(frame_idx)

        kps_raw = np.array(detection["keypoints_xyz"], dtype=np.float32)  # (17, 3) = [x, y, vis]

        # Handle unexpected shapes
        if kps_raw.ndim != 2 or kps_raw.shape[1] < 2:
            return None

        n_kps    = kps_raw.shape[0]
        kps_3d   = np.full((n_kps, 3), np.nan, dtype=np.float32)
        kps_conf = np.zeros(n_kps, dtype=np.float32)

        # Keypoint visibility / confidence is the 3rd column if present
        if kps_raw.shape[1] >= 3:
            kps_conf = kps_raw[:, 2]
        else:
            kps_conf[:] = 1.0   # assume all visible

        for k in range(n_kps):
            if kps_conf[k] < KP_CONF_THRESHOLD:
                continue   # leave as NaN

            u_f, v_f = float(kps_raw[k, 0]), float(kps_raw[k, 1])
            u_i, v_i = int(round(u_f)), int(round(v_f))

            # Boundary check
            if not (0 <= u_i < self.W and 0 <= v_i < self.H):
                continue

            if depth_m is not None:
                z = self._sample_depth(depth_m, u_i, v_i)
            else:
                # No depth frame available — fall back to unit depth
                # (SMPL fitter will handle scale ambiguity)
                z = 1.0

            if z < MIN_DEPTH_M or z > MAX_DEPTH_M:
                # bad depth — still back-project but flag conf
                kps_conf[k] *= 0.1
                z = 1.0   # unit depth fallback

            kps_3d[k] = self._backproject(u_f, v_f, z)

        return {
            "frame_index": frame_idx,
            "track_id":    detection.get("track_id_native", detection.get("person_id", -1)),
            "kps_3d":      kps_3d,        # (17, 3)
            "kps_conf":    kps_conf,       # (17,)
            "bbox_xywh":   detection["bbox_xywh"],
            "det_conf":    detection["conf"],
        }

    # ── Sequence ─────────────────────────────────────────────────────────────

    def lift_sequence(self, detections: list[dict]) -> list[dict]:
        """
        Lift a list of detections (already filtered to one track/person).
        Skips frames where lifting fails.
        """
        lifted = []
        missing_depth = 0

        for det in detections:
            result = self.lift_frame(det)
            if result is None:
                continue

            # Count frames with no depth file
            depth_path = self.depth_dir / f"depth_{det['frame_index']:06d}.npy"
            if not depth_path.exists():
                missing_depth += 1

            lifted.append(result)

        if missing_depth:
            print(f"  [DepthLifter] Warning: {missing_depth}/{len(detections)} frames had no depth file "
                  f"(unit-depth fallback used).")

        return lifted
# ─────────────────────────────────────────────────────────────────────────────
#  MONKEY PATCH — must be first, before any other imports
#  Fixes chumpy (SMPL dependency) on Python 3.11+ / NumPy 2.0+
# ─────────────────────────────────────────────────────────────────────────────
import inspect
import numpy as np

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

if not hasattr(np, 'int'):
    np.int     = np.int32
    np.bool    = getattr(np, 'bool_',   bool)
    np.float   = np.float64
    np.complex = np.complex128
    np.object  = getattr(np, 'object_', object)
    np.str     = np.str_
    np.unicode = np.str_

# ─────────────────────────────────────────────────────────────────────────────

"""
smpl_fitter.py
==============
Fits SMPL body model parameters to 3-D keypoints (lifted from depth) for each
frame, producing a temporally smooth animated sequence.

Dependencies
------------
    pip install smplx torch trimesh

SMPL model files
----------------
Download from https://smpl.is.tue.mpg.de  (register for free).
Place the .pkl files in the directory you pass as --smpl:
    SMPL_NEUTRAL.pkl  (or SMPL_FEMALE.pkl / SMPL_MALE.pkl)

COCO-17 → SMPL-24 joint mapping
---------------------------------
SMPL has 24 joints; COCO has 17 keypoints.  We map the 14 joints that
correspond well, and leave the rest unconstrained.

COCO idx : SMPL joint name   : SMPL idx
  0  nose          →  head        15
  5  l_shoulder    →  l_shoulder   16  (SMPL uses different indexing)
  6  r_shoulder    →  r_shoulder   17
  7  l_elbow       →  l_elbow      18
  8  r_elbow       →  r_elbow      19
  9  l_wrist       →  l_wrist      20
 10  r_wrist       →  r_wrist      21
 11  l_hip         →  l_hip         1
 12  r_hip         →  r_hip         2
 13  l_knee        →  l_knee        4
 14  r_knee        →  r_knee        5
 15  l_ankle       →  l_ankle       7
 16  r_ankle       →  r_ankle       8
"""

import os
import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False
    print("[SMPLFitter] Warning: smplx not installed. Run: pip install smplx")


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# COCO-17 index  →  SMPL-24 index
COCO_TO_SMPL = {
    0:  15,   # nose   → head
    5:  16,   # l_shoulder
    6:  17,   # r_shoulder
    7:  18,   # l_elbow
    8:  19,   # r_elbow
    9:  20,   # l_wrist
    10: 21,   # r_wrist
    11:  1,   # l_hip
    12:  2,   # r_hip
    13:  4,   # l_knee
    14:  5,   # r_knee
    15:  7,   # l_ankle
    16:  8,   # r_ankle
}

# Optimisation hyper-parameters
N_ITERS_SHAPE = 100   # Shape fitting  (run once on first frame)
N_ITERS_POSE  = 60    # Per-frame pose fitting
LR            = 3e-2

# Temporal smoothness weight (penalises large pose jumps between frames)
TEMPORAL_WEIGHT = 0.1


SMPLFrame = namedtuple("SMPLFrame", [
    "frame_index",
    "global_orient",   # (1, 3)   axis-angle
    "body_pose",       # (1, 69)  axis-angle  (23 joints × 3)
    "betas",           # (1, 10)  shape coefficients
    "transl",          # (1, 3)   root translation (metres)
    "vertices",        # (V, 3)   mesh vertices for rendering
    "joints",          # (J, 3)   SMPL joint positions
])


# ─────────────────────────────────────────────────────────────────────────────
#  SMPL FITTER
# ─────────────────────────────────────────────────────────────────────────────

class SMPLFitter:

    def __init__(self, smpl_model_dir: str, gender: str = "neutral", device: str | None = None):
        if not SMPLX_AVAILABLE:
            raise RuntimeError("smplx is required: pip install smplx")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[SMPLFitter] Using device: {self.device}")

        self.gender = gender
        self._build_model(smpl_model_dir, gender)

        # Shared shape betas (optimised once, then fixed)
        self.betas = nn.Parameter(torch.zeros(1, 10, device=self.device))

    # ── Model init ────────────────────────────────────────────────────────────

    def _build_model(self, model_dir: str, gender: str):
        """
        Loads SMPL via smplx.

        smplx.create with model_type='smpl' internally appends 'smpl/' to the
        path you give it, so the correct argument is the PARENT of the smpl/ folder.

        Both of these work as SMPL_DIR in run_pipeline.py:
            Models/          <- contains smpl/ subfolder
            Models/smpl/     <- we detect and step up one level automatically
        """
        model_dir = os.path.normpath(model_dir)

        # If user pointed directly at the smpl/ folder, step up one level
        # to avoid smplx constructing the path Models/smpl/smpl/
        if os.path.basename(model_dir).lower() == "smpl":
            model_dir = os.path.dirname(model_dir)

        self.smpl = smplx.create(
            model_dir,
            model_type="smpl",
            gender=gender,
            use_pca=False,
            batch_size=1,
        ).to(self.device)

        print(f"[SMPLFitter] SMPL model loaded (gender={gender})")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _kps_to_tensor(self, kps_3d: np.ndarray, kps_conf: np.ndarray):
        """
        Build target joint tensor and weight mask from COCO keypoints.
        Returns (target_joints [24,3], weights [24]) as torch tensors.
        """
        target = np.zeros((24, 3), dtype=np.float32)
        weights = np.zeros(24, dtype=np.float32)

        for coco_idx, smpl_idx in COCO_TO_SMPL.items():
            if not np.any(np.isnan(kps_3d[coco_idx])):
                conf = float(kps_conf[coco_idx])
                if conf > 0.0:
                    target[smpl_idx]  = kps_3d[coco_idx]
                    weights[smpl_idx] = conf

        target_t  = torch.tensor(target,  device=self.device)
        weights_t = torch.tensor(weights, device=self.device)
        return target_t, weights_t

    def _smpl_joints(self, global_orient, body_pose, betas, transl):
        output = self.smpl(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            return_verts=True,
        )
        return output.joints[:, :24, :], output.vertices

    def _root_from_kps(self, kps_3d: np.ndarray) -> np.ndarray:
        """Estimate root (pelvis) position as midpoint of hips."""
        l_hip = kps_3d[11] if not np.any(np.isnan(kps_3d[11])) else None
        r_hip = kps_3d[12] if not np.any(np.isnan(kps_3d[12])) else None

        if l_hip is not None and r_hip is not None:
            return ((l_hip + r_hip) / 2.0).astype(np.float32)
        for idx in [11, 12, 5, 6]:
            if not np.any(np.isnan(kps_3d[idx])):
                return kps_3d[idx].astype(np.float32)
        valid = kps_3d[~np.isnan(kps_3d).any(axis=1)]
        return valid.mean(axis=0).astype(np.float32) if len(valid) else np.zeros(3, dtype=np.float32)

    # ── Shape pre-fitting (one-time) ──────────────────────────────────────────

    def fit_shape(self, frames_3d: list[dict]):
        """
        Run a short optimisation over the first N frames to estimate body shape
        (betas).  Shape is assumed constant across the sequence.
        """
        print("  Fitting body shape (betas) on first frames …")
        n_shape_frames = min(30, len(frames_3d))
        shape_frames   = frames_3d[:n_shape_frames]

        self.betas   = nn.Parameter(torch.zeros(1, 10, device=self.device))
        orient_init  = nn.Parameter(torch.zeros(1,  3, device=self.device))
        pose_init    = nn.Parameter(torch.zeros(1, 69, device=self.device))
        transl_init  = nn.Parameter(torch.zeros(1,  3, device=self.device))

        optimizer = torch.optim.Adam(
            [self.betas, orient_init, pose_init, transl_init], lr=LR
        )

        for _ in range(N_ITERS_SHAPE):
            total_loss = torch.tensor(0.0, device=self.device)
            for frame in shape_frames:
                target_t, weights_t = self._kps_to_tensor(frame["kps_3d"], frame["kps_conf"])
                joints, _ = self._smpl_joints(orient_init, pose_init, self.betas, transl_init)
                diff = (joints[0] - target_t) ** 2  # (24, 3)
                per_joint = diff.sum(dim=-1)         # (24,)
                total_loss = total_loss + (per_joint * weights_t).mean()

            # Shape regularisation
            total_loss = total_loss + 0.01 * (self.betas ** 2).sum()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Detach betas — kept fixed for per-frame pose fitting
        self.betas = nn.Parameter(self.betas.detach())
        print(f"  Shape fitting done. betas: {self.betas.data.cpu().numpy().round(3)}")

    # ── Per-frame pose fitting ────────────────────────────────────────────────

    def fit_frame(
        self,
        frame_data: dict,
        prev_orient: torch.Tensor | None = None,
        prev_pose:   torch.Tensor | None = None,
    ) -> SMPLFrame:
        """Fit pose parameters for a single frame."""

        target_t, weights_t = self._kps_to_tensor(frame_data["kps_3d"], frame_data["kps_conf"])

        # Initialise from previous frame for temporal coherence
        orient_init = prev_orient.clone().detach() if prev_orient is not None \
                      else torch.zeros(1, 3, device=self.device)
        pose_init   = prev_pose.clone().detach()   if prev_pose   is not None \
                      else torch.zeros(1, 69, device=self.device)

        root_np    = self._root_from_kps(frame_data["kps_3d"])
        transl_t   = nn.Parameter(torch.tensor(root_np[None], device=self.device))
        orient_p   = nn.Parameter(orient_init)
        pose_p     = nn.Parameter(pose_init)

        optimizer  = torch.optim.Adam([orient_p, pose_p, transl_t], lr=LR)

        for _ in range(N_ITERS_POSE):
            joints, _ = self._smpl_joints(orient_p, pose_p, self.betas.detach(), transl_t)
            diff       = (joints[0] - target_t) ** 2
            joint_loss = (diff.sum(dim=-1) * weights_t).mean()

            # Pose regularisation (keep close to rest pose)
            reg_loss = 0.005 * (pose_p ** 2).mean()

            # Temporal smoothness
            smooth_loss = torch.tensor(0.0, device=self.device)
            if prev_orient is not None:
                smooth_loss = smooth_loss + TEMPORAL_WEIGHT * ((orient_p - prev_orient) ** 2).mean()
            if prev_pose is not None:
                smooth_loss = smooth_loss + TEMPORAL_WEIGHT * ((pose_p - prev_pose) ** 2).mean()

            loss = joint_loss + reg_loss + smooth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final forward pass for vertices
        with torch.no_grad():
            joints_out, verts_out = self._smpl_joints(orient_p, pose_p, self.betas.detach(), transl_t)

        return SMPLFrame(
            frame_index  = frame_data["frame_index"],
            global_orient= orient_p.detach().cpu().numpy(),
            body_pose    = pose_p.detach().cpu().numpy(),
            betas        = self.betas.detach().cpu().numpy(),
            transl       = transl_t.detach().cpu().numpy(),
            vertices     = verts_out[0].detach().cpu().numpy(),
            joints       = joints_out[0].detach().cpu().numpy(),
        ), orient_p.detach(), pose_p.detach()

    # ── Sequence fitting ──────────────────────────────────────────────────────

    def fit_sequence(self, frames_3d: list[dict]) -> list[SMPLFrame]:
        """Fit SMPL to every frame in the sequence."""
        self.fit_shape(frames_3d)

        results    = []
        prev_orient = None
        prev_pose   = None

        for i, frame in enumerate(frames_3d):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Frame {i+1}/{len(frames_3d)}  (frame_index={frame['frame_index']})")

            smpl_frame, prev_orient, prev_pose = self.fit_frame(
                frame, prev_orient, prev_pose
            )
            results.append(smpl_frame)

        return results

    # ── Save / load ───────────────────────────────────────────────────────────

    @staticmethod
    def save_sequence(seq: list[SMPLFrame], path: str):
        data = [frame._asdict() for frame in seq]
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[SMPLFitter] Sequence saved → {path}")

    @staticmethod
    def load_sequence(path: str) -> list[SMPLFrame]:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return [SMPLFrame(**d) for d in data]
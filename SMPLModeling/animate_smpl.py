"""
animate_smpl.py
===============
Renders the fitted SMPL sequence as an interactive 3-D animation using
pyrender + trimesh, and can also export a video (MP4) or per-frame images.

Dependencies
------------
    pip install pyrender trimesh smplx torch

Usage (standalone — for replaying a saved .pkl)
-----------------------------------------------
    python animate_smpl.py \
        --pkl  output_dir/track_1_smpl_sequence.pkl \
        --smpl path/to/smpl_models/ \
        --gender neutral
"""

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np

try:
    import trimesh
    import pyrender
    RENDER_AVAILABLE = True
except ImportError:
    RENDER_AVAILABLE = False
    print("[SMPLAnimator] Warning: pyrender/trimesh not installed. Run: pip install pyrender trimesh")

try:
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from smpl_fitter import SMPLFrame, SMPLFitter


# SMPL face indices are fixed — cache them once
_SMPL_FACES: np.ndarray | None = None


def get_smpl_faces(smpl_model_dir: str, gender: str) -> np.ndarray:
    global _SMPL_FACES
    if _SMPL_FACES is None:
        import torch
        import smplx
        smpl = smplx.create(smpl_model_dir, model_type="smpl", gender=gender,
                            use_pca=False, batch_size=1)
        _SMPL_FACES = smpl.faces.astype(np.int32)
    return _SMPL_FACES


# ─────────────────────────────────────────────────────────────────────────────
#  ANIMATOR
# ─────────────────────────────────────────────────────────────────────────────

class SMPLAnimator:

    def __init__(self, smpl_model_dir: str, gender: str = "neutral",
                 fps: float = 30.0, save_video: str | None = None):
        if not RENDER_AVAILABLE:
            raise RuntimeError("pyrender + trimesh required: pip install pyrender trimesh")

        self.fps            = fps
        self.save_video     = save_video
        self.smpl_model_dir = smpl_model_dir
        self.gender         = gender

    # ── Mesh builder ─────────────────────────────────────────────────────────

    def _make_mesh(self, vertices: np.ndarray, faces: np.ndarray,
                   color: tuple = (0.65, 0.74, 0.86, 1.0)) -> "pyrender.Mesh":
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces,
                                   vertex_colors=None, process=False)
        tri_mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi, [1, 0, 0]))   # flip Y to match OpenGL convention

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.1,
            roughnessFactor=0.7,
            alphaMode="OPAQUE",
        )
        return pyrender.Mesh.from_trimesh(tri_mesh, material=material, smooth=True)

    # ── Scene builder ─────────────────────────────────────────────────────────

    def _build_scene(self) -> tuple:
        """Create a pyrender scene with lights and camera."""
        scene = pyrender.Scene(bg_color=[0.12, 0.12, 0.15, 1.0],
                               ambient_light=[0.2, 0.2, 0.2])

        # Key light
        key_light = pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=3.0)
        kl_pose   = np.eye(4)
        kl_pose[:3, :3] = trimesh.transformations.rotation_matrix(
            np.radians(-45), [1, 0, 0])[:3, :3]
        scene.add(key_light, pose=kl_pose)

        # Fill light
        fill_light = pyrender.DirectionalLight(color=[0.6, 0.7, 1.0], intensity=1.5)
        fl_pose    = np.eye(4)
        fl_pose[:3, :3] = trimesh.transformations.rotation_matrix(
            np.radians(30), [0, 1, 0])[:3, :3]
        scene.add(fill_light, pose=fl_pose)

        # Camera — positioned 3m back, looking forward
        camera       = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=16/9)
        camera_pose  = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.8],   # slight upward offset
            [0, 0, 1, 3.0],   # 3m back
            [0, 0, 0, 1],
        ], dtype=np.float32)
        camera_node = scene.add(camera, pose=camera_pose)

        return scene, camera_node

    # ── Interactive playback ──────────────────────────────────────────────────

    def play(self, seq: list[SMPLFrame]):
        """
        Play the sequence in an interactive pyrender window.
        Press Q or close the window to stop.
        """
        faces   = get_smpl_faces(self.smpl_model_dir, self.gender)
        scene, _= self._build_scene()

        # Determine a stable camera look-at based on median root position
        roots = np.array([f.transl[0] for f in seq])
        centre = np.median(roots, axis=0)

        # Build viewer
        viewer = pyrender.Viewer(
            scene,
            use_raymond_lighting=False,
            viewport_size=(1280, 720),
            run_in_thread=True,
        )

        mesh_node = None
        frame_dt  = 1.0 / self.fps

        print(f"[SMPLAnimator] Playing {len(seq)} frames at {self.fps:.1f} fps …")
        print("               Close the window to stop.")

        try:
            for smpl_frame in seq:
                t0 = time.time()

                new_mesh = self._make_mesh(smpl_frame.vertices, faces)

                with viewer.render_lock:
                    if mesh_node is not None:
                        scene.remove_node(mesh_node)
                    mesh_node = scene.add(new_mesh)

                # Frame-rate cap
                elapsed = time.time() - t0
                wait    = frame_dt - elapsed
                if wait > 0:
                    time.sleep(wait)

                if not viewer.is_active:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            viewer.close_external()

    # ── Offline render → video ────────────────────────────────────────────────

    def render_to_video(self, seq: list[SMPLFrame], output_path: str,
                        width: int = 1280, height: int = 720):
        """
        Render each frame offscreen and write an MP4.
        Requires opencv-python: pip install opencv-python
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("opencv-python required: pip install opencv-python")

        faces    = get_smpl_faces(self.smpl_model_dir, self.gender)
        scene, _ = self._build_scene()

        renderer = pyrender.OffscreenRenderer(width, height)
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        print(f"[SMPLAnimator] Rendering {len(seq)} frames to {output_path} …")

        for i, smpl_frame in enumerate(seq):
            mesh = self._make_mesh(smpl_frame.vertices, faces)
            node = scene.add(mesh)

            color, _ = renderer.render(scene)
            bgr      = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

            scene.remove_node(node)

            if (i + 1) % 30 == 0:
                print(f"  Rendered {i+1}/{len(seq)} frames")

        writer.release()
        renderer.delete()
        print(f"[SMPLAnimator] Video saved → {output_path}")

    # ── Export per-frame meshes (.obj) ────────────────────────────────────────

    def export_meshes(self, seq: list[SMPLFrame], out_dir: str):
        """Save each frame as a .obj file for use in Blender, MeshLab, etc."""
        faces = get_smpl_faces(self.smpl_model_dir, self.gender)
        os.makedirs(out_dir, exist_ok=True)

        for smpl_frame in seq:
            mesh = trimesh.Trimesh(
                vertices=smpl_frame.vertices,
                faces=faces,
                process=False,
            )
            path = os.path.join(out_dir, f"frame_{smpl_frame.frame_index:06d}.obj")
            mesh.export(path)

        print(f"[SMPLAnimator] {len(seq)} .obj files saved → {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Replay a saved SMPL sequence")
    p.add_argument("--pkl",    required=True, help="Path to _smpl_sequence.pkl")
    p.add_argument("--smpl",   required=True, help="SMPL model directory")
    p.add_argument("--gender", default="neutral", choices=["neutral","male","female"])
    p.add_argument("--fps",    type=float, default=30.0)
    p.add_argument("--video",  default=None, help="Save to MP4 instead of interactive viewer")
    p.add_argument("--meshes", default=None, help="Export per-frame .obj files to this directory")
    args = p.parse_args()

    seq = SMPLFitter.load_sequence(args.pkl)
    print(f"Loaded {len(seq)} frames from {args.pkl}")

    animator = SMPLAnimator(args.smpl, gender=args.gender, fps=args.fps)

    if args.meshes:
        animator.export_meshes(seq, args.meshes)
    elif args.video:
        animator.render_to_video(seq, args.video)
    else:
        animator.play(seq)


if __name__ == "__main__":
    main()
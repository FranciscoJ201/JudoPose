"""
run_pipeline.py
===============
Entry point for lifting YOLO pose keypoints into 3-D and fitting an animated
SMPL body model for a single tracked person.

Edit the CONFIG block below, then run:  python run_pipeline.py
"""

import json
import os
import sys

from depth_lift   import DepthLifter
from smpl_fitter  import SMPLFitter
from animate_smpl import SMPLAnimator


# ─────────────────────────────────────────────────────────────────────────────
#  ✏️  CONFIG — edit these paths before running
# ─────────────────────────────────────────────────────────────────────────────

JSON_PATH   = "my_video_pose_detection.json"   # output from poseestimation.py
DEPTH_DIR   = "realsense_cam_0/depth_frames/"  # folder of depth_XXXXXX.npy files
INTR_PATH   = "realsense_cam_0/intrinsics.json"
SMPL_DIR    = "smpl_models/"                   # folder containing smpl/ subdir with .pkl files
OUTPUT_DIR  = "smpl_output/"

TRACK_ID    = 1        # track_id_native to follow — set to None to list all IDs and exit
GENDER      = "neutral"   # "neutral" | "male" | "female"
START_FRAME = 0        # first frame to process
END_FRAME   = None     # last frame to process (None = all frames)
DEVICE      = None     # None = auto-detect CUDA, or "cpu" / "cuda"
NO_RENDER   = False    # set True to skip the interactive viewer
    p.add_argument("--device",      default=None,   help="'cpu' or 'cuda'")
    p.add_argument("--no-render",   action="store_true")
    p.add_argument("--gender",      default="neutral", choices=["neutral","male","female"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_detections(json_path: str) -> list[dict]:
    with open(json_path, "r") as f:
        return json.load(f)


def list_tracks(detections: list[dict]):
    ids = sorted(set(d.get("track_id_native", d.get("person_id", -1)) for d in detections))
    print("Track IDs found in JSON:")
    for tid in ids:
        count = sum(1 for d in detections if d.get("track_id_native", d.get("person_id")) == tid)
        print(f"  ID {tid:4d}  —  {count} detections")


def filter_detections(detections, track_id, start_frame, end_frame):
    """Return detections for *one person*, sorted by frame_index."""
    out = [
        d for d in detections
        if d.get("track_id_native", d.get("person_id")) == track_id
        and (end_frame is None or d["frame_index"] <= end_frame)
        and d["frame_index"] >= start_frame
    ]
    return sorted(out, key=lambda d: d["frame_index"])


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load detections
    print(f"Loading detections from {JSON_PATH} …")
    detections = load_detections(JSON_PATH)
    print(f"  {len(detections)} total detections loaded.")

    # If TRACK_ID is None, list available IDs and exit
    if TRACK_ID is None:
        list_tracks(detections)
        print("\nSet TRACK_ID in the CONFIG block above, then re-run.")
        sys.exit(0)

    # 2. Filter to one person
    person_dets = filter_detections(detections, TRACK_ID, START_FRAME, END_FRAME)
    if not person_dets:
        print(f"[ERROR] No detections found for track ID {TRACK_ID} in the given frame range.")
        sys.exit(1)
    print(f"  {len(person_dets)} detections for track ID {TRACK_ID}.")

    # 3. Lift keypoints to 3-D
    print("\nLifting 2-D keypoints → 3-D using depth maps …")
    lifter    = DepthLifter(INTR_PATH, DEPTH_DIR)
    frames_3d = lifter.lift_sequence(person_dets)
    print(f"  Lifted {len(frames_3d)} frames.")

    # 4. Fit SMPL to every frame
    print("\nFitting SMPL model per frame …")
    fitter   = SMPLFitter(SMPL_DIR, gender=GENDER, device=DEVICE)
    smpl_seq = fitter.fit_sequence(frames_3d)
    print(f"  Fitting complete for {len(smpl_seq)} frames.")

    # 5. Save results
    pkl_path = os.path.join(OUTPUT_DIR, f"track_{TRACK_ID}_smpl_sequence.pkl")
    fitter.save_sequence(smpl_seq, pkl_path)
    print(f"\nSaved SMPL sequence → {pkl_path}")

    # 6. Animate / render
    if not NO_RENDER:
        print("\nLaunching animated viewer …  (close window to exit)")
        animator = SMPLAnimator(SMPL_DIR, gender=GENDER)
        animator.play(smpl_seq)
    else:
        print("NO_RENDER=True: skipping viewer.")

    print("\nDone.")


if __name__ == "__main__":
    main()
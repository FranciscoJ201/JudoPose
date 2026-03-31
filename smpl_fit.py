import numpy as np
import json
import torch
import smplx
import os
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────
# THE ANATOMY MAP: COCO 17 (YOLO) to SMPL 24
# SMPL has 24 primary joints. YOLO outputs 17. 
# We map the exact index of the YOLO joint to its corresponding SMPL joint.
# ─────────────────────────────────────────────────────────────────────────
COCO_TO_SMPL = {
    # COCO Index : SMPL Index
    15: 7,   # Left Ankle
    13: 4,   # Left Knee
    11: 1,   # Left Hip
    12: 2,   # Right Hip
    14: 5,   # Right Knee
    16: 8,   # Right Ankle
    5:  16,  # Left Shoulder
    7:  18,  # Left Elbow
    9:  20,  # Left Wrist
    6:  17,  # Right Shoulder
    8:  19,  # Right Elbow
    10: 21,  # Right Wrist
    # Note: We omit facial features (eyes/ears) for raw body kinematics, 
    # relying on the neck/shoulders to orient the torso.
}

class SMPLFitter:
    def __init__(self, smpl_model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the PyTorch SMPL model and the optimization engine.
        
        Arguments:
        - smpl_model_path (str): Path to the folder containing SMPL_NEUTRAL.pkl
        - device (str): 'cuda' for GPU acceleration or 'cpu'
        """
        self.device = device
        print(f"[SMPL Fitter] Booting optimization engine on: {self.device.upper()}")
        
        # Load the SMPL body model
        self.smpl = smplx.create(
            model_path=smpl_model_path, 
            model_type='smpl',
            gender='neutral', 
            ext='pkl'
        ).to(self.device)

        # Pre-build mapping arrays for fast tensor indexing
        self.coco_idx = list(COCO_TO_SMPL.keys())
        self.smpl_idx = list(COCO_TO_SMPL.values())

    def fit_frame(self, target_3d_joints, init_pose=None, init_shape=None, init_trans=None, iterations=100):
        """
        Runs gradient descent to warp the SMPL mesh to match the 3D target dots.
        
        Arguments:
        - target_3d_joints: (17, 3) numpy array of triangulated [X, Y, Z] points.
        - init_pose:        (1, 72) tensor of starting joint angles.
        - init_shape:       (1, 10) tensor of starting body proportions.
        - init_trans:       (1, 3) tensor of starting global pelvis location.
        - iterations:       (int) number of gradient descent steps.
        
        Returns:
        - tuple of optimized (pose, shape, translation) tensors.
        """
        # Convert targets to PyTorch tensors
        target_pts = torch.tensor(target_3d_joints, dtype=torch.float32, device=self.device)
        
        # Create a mask to completely ignore NaN values (blocked cameras)
        valid_mask = ~torch.isnan(target_pts[:, 0])
        
        if not valid_mask.any():
            return init_pose, init_shape, init_trans # Skip empty frames

        # Initialize trainable parameters
        # If we have data from the previous frame, we use it as a "warm start" 
        # to ensure temporal smoothness and vastly speed up convergence.
        body_pose = (init_pose.clone().detach() if init_pose is not None else torch.zeros(1, 69, device=self.device))
        global_orient = torch.zeros(1, 3, device=self.device)
        betas = (init_shape.clone().detach() if init_shape is not None else torch.zeros(1, 10, device=self.device))
        transl = (init_trans.clone().detach() if init_trans is not None else torch.zeros(1, 3, device=self.device))

        body_pose.requires_grad_(True)
        global_orient.requires_grad_(True)
        betas.requires_grad_(True)
        transl.requires_grad_(True)

        optimizer = torch.optim.Adam([body_pose, global_orient, betas, transl], lr=0.05)

        for _ in range(iterations):
            optimizer.zero_grad()

            # Forward pass: Generate the 3D mesh and internal joints
            output = self.smpl(
                body_pose=body_pose,
                global_orient=global_orient,
                betas=betas,
                transl=transl
            )
            
            # Extract only the 24 internal joints
            smpl_joints = output.joints[0, :24, :]

            # Filter joints based on our COCO mapping and valid tracking data
            pred_mapped = smpl_joints[self.smpl_idx][valid_mask[self.coco_idx]]
            targ_mapped = target_pts[self.coco_idx][valid_mask[self.coco_idx]]

            # 1. Data Loss: Distance between SMPL bones and YOLO dots
            loss_data = torch.nn.functional.mse_loss(pred_mapped, targ_mapped)

            # 2. Shape Prior: Penalize extreme body morphing (keep it human)
            loss_shape = torch.sum(betas ** 2)

            # 3. Pose Prior: Penalize extreme joint angles
            loss_pose = torch.sum(body_pose ** 2)

            # Calculate total loss with weighting factors
            loss_total = loss_data + (0.001 * loss_shape) + (0.01 * loss_pose)

            # Backward pass & Optimize
            loss_total.backward()
            optimizer.step()

        # Combine global orientation and body pose into the standard 72D format
        full_pose = torch.cat([global_orient, body_pose], dim=1)
        
        return full_pose.detach(), betas.detach(), transl.detach()


def process_smpl_kinematics(json_path, smpl_model_path, output_path):
    """
    Orchestrates the conversion of floating 3D coordinates into a physical 
    biomechanical mesh over the entire timeline.
    
    Arguments:
    - json_path:       (str) Path to final_3d.json (Output of Stage 5)
    - smpl_model_path: (str) Path to folder with SMPL_NEUTRAL.pkl
    - output_path:     (str) Path to save the optimized kinematics (.npz)
    """
    print("Loading triangulated 3D data...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_frames = len(data)
    fitter = SMPLFitter(smpl_model_path=smpl_model_path)

    # Output storage arrays
    out_poses  = np.zeros((num_frames, 72))
    out_shapes = np.zeros((num_frames, 10))
    out_trans  = np.zeros((num_frames, 3))

    print(f"Executing PyTorch SMPL Optimization on {num_frames} frames...")
    
    # State variables for temporal smoothing (warm starts)
    prev_pose, prev_shape, prev_trans = None, None, None

    for i in tqdm(range(num_frames), desc="Fitting Mesh"):
        frame_data = data[i]['keypoints_3d']
        target_pts = np.array([kp['xyz'] for kp in frame_data])

        # Run the optimizer for this frame
        pose, shape, trans = fitter.fit_frame(
            target_3d_joints=target_pts,
            init_pose=prev_pose[:, 3:] if prev_pose is not None else None, # Omit global orient
            init_shape=prev_shape,
            init_trans=prev_trans,
            iterations=80 # Optimized iteration count for temporal processing
        )

        # Store results
        out_poses[i]  = pose.cpu().numpy()
        out_shapes[i] = shape.cpu().numpy()
        out_trans[i]  = trans.cpu().numpy()

        # Update warm start variables
        prev_pose, prev_shape, prev_trans = pose, shape, trans

    # Save the biomechanical parameters
    np.savez(
        output_path, 
        poses=out_poses, 
        shapes=out_shapes, 
        trans=out_trans
    )
    print(f"Done! Kinematics saved to {output_path}")

if __name__ == "__main__":
    pass
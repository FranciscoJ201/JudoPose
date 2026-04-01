import numpy as np
import torch
import smplx
import trimesh
import time

def visualize_kinematics(npz_path, smpl_model_path, fps=30):
    """
    Reads SMPL kinematics from an .npz file and renders a real-time 3D 
    animation of the human mesh using Trimesh.

    Arguments:
    - npz_path:        (str) Path to the saved kinematics .npz file.
    - smpl_model_path: (str) Path to the folder containing SMPL_NEUTRAL.pkl.
    - fps:             (int) Target frames per second for the playback window.
    """
    print(f"[Visualizer] Loading kinematics from {npz_path}...")
    data = np.load(npz_path)
    poses = data['poses']   
    shapes = data['shapes'] 
    trans = data['trans']   

    num_frames = poses.shape[0]

    print("[Visualizer] Booting SMPL model (CPU mode for rendering)...")
    smpl = smplx.create(
        model_path=smpl_model_path, 
        model_type='smpl',
        gender='neutral', 
        ext='pkl'
    )

    # ─────────────────────────────────────────────────────────────────────────
    # INITIALIZE TRIMESH SCENE & BASE MESH
    # ─────────────────────────────────────────────────────────────────────────
    print("[Visualizer] Initializing 3D Render Window...")
    
    # Generate the frame 0 mesh to establish the geometry
    with torch.no_grad():
        init_out = smpl(
            global_orient=torch.tensor(poses[0:1, :3], dtype=torch.float32),
            body_pose=torch.tensor(poses[0:1, 3:], dtype=torch.float32),
            betas=torch.tensor(shapes[0:1, :], dtype=torch.float32),
            transl=torch.tensor(trans[0:1, :], dtype=torch.float32)
        )
    
    # Create the Trimesh object
    mesh = trimesh.Trimesh(
        vertices=init_out.vertices[0].numpy(), 
        faces=smpl.faces,
        process=False # Prevent Trimesh from altering our exact SMPL vertex order
    )
    
    # Set a neutral visual color (RGBA format)
    mesh.visual.vertex_colors = [150, 175, 200, 255]

    # Create the scene and add our mesh
    scene = trimesh.Scene([mesh])

    # ─────────────────────────────────────────────────────────────────────────
    # THE PLAYBACK CALLBACK LOOP
    # ─────────────────────────────────────────────────────────────────────────
    # We use a dictionary to maintain state inside the callback function
    playback_state = {
        'frame': 0,
        'last_update_time': time.time(),
        'frame_duration': 1.0 / fps
    }

    def update_scene(scene_obj):
        """
        Callback function executed continuously by the Trimesh viewer.
        Updates the mesh vertices to the next frame in the timeline.
        """
        current_time = time.time()
        
        # Rate-limit the playback to match the target FPS
        if (current_time - playback_state['last_update_time']) < playback_state['frame_duration']:
            return
            
        frame_idx = playback_state['frame']
        
        # Extract current frame parameters
        global_orient = torch.tensor(poses[frame_idx:frame_idx+1, :3], dtype=torch.float32)
        body_pose = torch.tensor(poses[frame_idx:frame_idx+1, 3:], dtype=torch.float32)
        betas = torch.tensor(shapes[frame_idx:frame_idx+1, :], dtype=torch.float32)
        transl = torch.tensor(trans[frame_idx:frame_idx+1, :], dtype=torch.float32)

        # Generate the new vertices for this frame
        with torch.no_grad():
            output = smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl
            )
        
        # Update the geometry in place
        mesh.vertices = output.vertices[0].numpy()
        
        # Advance the frame counter, loop back to 0 if at the end
        playback_state['frame'] = (frame_idx + 1) % num_frames
        playback_state['last_update_time'] = current_time

    print(f"[Visualizer] Playing {num_frames} frames at {fps} FPS. Close the window to exit.")
    
    # Launch the interactive window with the callback attached
    scene.show(callback=update_scene, smooth=False)
    print("[Visualizer] Playback complete.")

if __name__ == "__main__":
    pass
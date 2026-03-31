import numpy as np
import torch
import smplx
import open3d as o3d
import time

def visualize_kinematics(npz_path, smpl_model_path, fps=30):
    """
    Reads SMPL kinematics from an .npz file and renders a real-time 3D 
    animation of the human mesh using Open3D.

    Arguments:
    - npz_path:        (str) Path to the saved kinematics .npz file.
    - smpl_model_path: (str) Path to the folder containing SMPL_NEUTRAL.pkl.
    - fps:             (int) Target frames per second for the playback window.
    """
    print(f"[Visualizer] Loading kinematics from {npz_path}...")
    data = np.load(npz_path)
    poses = data['poses']   # Shape: (num_frames, 72)
    shapes = data['shapes'] # Shape: (num_frames, 10)
    trans = data['trans']   # Shape: (num_frames, 3)

    num_frames = poses.shape[0]

    print("[Visualizer] Booting SMPL model (CPU mode for rendering)...")
    # For visualization, CPU is perfectly fast enough and avoids VRAM overhead
    smpl = smplx.create(
        model_path=smpl_model_path, 
        model_type='smpl',
        gender='neutral', 
        ext='pkl'
    )

    # ─────────────────────────────────────────────────────────────────────────
    # INITIALIZE OPEN3D WINDOW & MESH
    # ─────────────────────────────────────────────────────────────────────────
    print("[Visualizer] Initializing 3D Render Window...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Judo Biomechanics - SMPL Playback", width=1280, height=720)

    # Create an empty triangle mesh object
    mesh = o3d.geometry.TriangleMesh()
    
    # The faces (triangles) never change across frames, so we set them once
    mesh.triangles = o3d.utility.Vector3iVector(smpl.faces)
    
    # Paint the mesh a solid color (e.g., a neutral grey/blue)
    mesh.paint_uniform_color([0.6, 0.7, 0.8])
    mesh.compute_vertex_normals()

    # Add the mesh to the visualizer
    vis.add_geometry(mesh)

    # Optional: Add a coordinate frame to represent the global Judo mat origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)

    # ─────────────────────────────────────────────────────────────────────────
    # THE PLAYBACK LOOP
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[Visualizer] Playing {num_frames} frames at {fps} FPS. Close the window to exit.")
    
    sleep_time = 1.0 / fps

    for i in range(num_frames):
        # Slice out the current frame's parameters and convert to tensors
        # SMPL requires shape (1, N) for forward passes
        global_orient = torch.tensor(poses[i:i+1, :3], dtype=torch.float32)
        body_pose = torch.tensor(poses[i:i+1, 3:], dtype=torch.float32)
        betas = torch.tensor(shapes[i:i+1, :], dtype=torch.float32)
        transl = torch.tensor(trans[i:i+1, :], dtype=torch.float32)

        # Generate the 3D mesh for this specific frame
        with torch.no_grad():
            output = smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl
            )
        
        # Extract the 6890 vertices and convert them back to a standard numpy array
        vertices = output.vertices[0].numpy()

        # Update the Open3D mesh object with the new vertex positions
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.compute_vertex_normals() # Recompute lighting shadows

        # Refresh the window
        vis.update_geometry(mesh)
        
        # Allow the window to catch events (mouse rotations, closing)
        if not vis.poll_events():
            print("[Visualizer] Window closed by user.")
            break
            
        vis.update_renderer()

        # Pause to maintain accurate FPS playback speed
        time.sleep(sleep_time)

    print("[Visualizer] Playback complete.")
    vis.destroy_window()

if __name__ == "__main__":
    pass
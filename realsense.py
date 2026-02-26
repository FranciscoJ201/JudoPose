import pyrealsense2 as rs
import numpy as np
import json

def export_realsense_intrinsics(output_file="intrinsic_realsense.json"):
    """
    Connects to an Intel RealSense camera, extracts its factory-calibrated 
    intrinsic parameters, and saves them to a JSON file.
    """
    print("Connecting to RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable the color stream (this is the lens YOLO will be looking through)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    try:
        # Start the camera pipeline
        profile = pipeline.start(config)
        
        # Grab the color stream profile
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        print("\n--- Factory Intrinsics Retrieved ---")
        print(f"Resolution: {intrinsics.width}x{intrinsics.height}")
        print(f"Focal Length (fx, fy): {intrinsics.fx:.2f}, {intrinsics.fy:.2f}")
        print(f"Principal Point (ppx, ppy): {intrinsics.ppx:.2f}, {intrinsics.ppy:.2f}")
        
        # 1. Build the Camera Matrix (K)
        # K = [[fx,  0, ppx],
        #      [ 0, fy, ppy],
        #      [ 0,  0,   1]]
        K = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # 2. Extract Distortion Coefficients (D)
        # RealSense typically uses a 5-parameter Brown-Conrady model
        D = np.array(intrinsics.coeffs).reshape(5, 1)
        
        # 3. Save to JSON in our standard format
        calibration_data = {
            "camera_model": "standard", # RealSense uses standard radial distortion
            "reprojection_error": 0.0,  # Factory calibration is assumed perfect baseline
            "camera_matrix": K.tolist(),
            "distortion_coefficients": D.tolist(),
            "image_size": [intrinsics.width, intrinsics.height]
        }
        
        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=4)
            
        print(f"\nSUCCESS! Saved factory calibration to {output_file}")
        
    except Exception as e:
        print(f"Error communicating with RealSense: {e}")
        
    finally:
        pipeline.stop()

if __name__ == "__main__":
    export_realsense_intrinsics()
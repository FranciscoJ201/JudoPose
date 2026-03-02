import cv2
import numpy as np
import json
import os

# Global variable to store the points the user clicks
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    """Captures mouse clicks to define the mat corners."""
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((float(x), float(y)))
            img_copy = param.copy()
            
            # Draw the points and lines as the user clicks
            for i, p in enumerate(clicked_points):
                cv2.circle(img_copy, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
                cv2.putText(img_copy, str(i+1), (int(p[0])+10, int(p[1])-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if i > 0:
                    cv2.line(img_copy, (int(clicked_points[i-1][0]), int(clicked_points[i-1][1])), 
                             (int(p[0]), int(p[1])), (0, 255, 255), 2)
            
            # Close the loop if 4 points are clicked
            if len(clicked_points) == 4:
                cv2.line(img_copy, (int(clicked_points[3][0]), int(clicked_points[3][1])), 
                         (int(clicked_points[0][0]), int(clicked_points[0][1])), (0, 255, 255), 2)
                print("\n4 points recorded! Press any key to continue...")

            cv2.imshow("Click the 4 corners of the Judo Mat", img_copy)

def calibrate_extrinsics_pnp(image_path, intrinsic_json, output_json, mat_size_meters=8.0):
    """
    Calculates the 3D position (Extrinsics) of a camera by clicking the corners of a known object (the Judo mat).
    """
    global clicked_points
    clicked_points = [] # Reset for multiple runs

    # --- 1. Load Intrinsic Data ---
    if not os.path.exists(intrinsic_json):
        print(f"Error: Could not find {intrinsic_json}")
        return

    with open(intrinsic_json, 'r') as f:
        data = json.load(f)
    
    K = np.array(data["camera_matrix"])
    D = np.array(data["distortion_coefficients"])
    camera_model = data.get("camera_model", "standard")

    # --- 2. Load and Undistort Image ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    h, w = img.shape[:2]

    print("Undistorting image for precise clicking...")
    if camera_model == "fisheye":
        # Calculate optimal camera matrix for flat viewing (Legacy GoPro math)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.5)
        undistorted_img = cv2.fisheye.undistortImage(img, K, D, Knew=new_K)
    else:
        # Both "standard" (RealSense) and "fisheye_rational" (New GoPro math) use standard undistort
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
        undistorted_img = cv2.undistort(img, K, D, None, new_K)

    # --- 3. UI: User Clicks the 4 Corners ---
    print("\n--- INSTRUCTIONS ---")
    print("Click the 4 corners of the Judo combat area in exactly this order:")
    print("1. Top-Left")
    print("2. Top-Right")
    print("3. Bottom-Right")
    print("4. Bottom-Left")
    
    cv2.imshow("Click the 4 corners of the Judo Mat", undistorted_img)
    cv2.setMouseCallback("Click the 4 corners of the Judo Mat", mouse_callback, undistorted_img)
    
    # Wait until the user clicks 4 points and presses a key
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicked_points) != 4:
        print("Error: You did not click exactly 4 points. Aborting.")
        return

    # --- 4. The Math (SolvePnP) ---
    image_points = np.array(clicked_points, dtype=np.float32)

    # Define the literal 3D coordinates of the mat corners in meters.
    # Center of the mat is (0,0,0). Z is 0 (the floor).
    half_mat = mat_size_meters / 2.0
    object_points = np.array([
        [-half_mat,  half_mat, 0.0], # Top-Left
        [ half_mat,  half_mat, 0.0], # Top-Right
        [ half_mat, -half_mat, 0.0], # Bottom-Right
        [-half_mat, -half_mat, 0.0]  # Bottom-Left
    ], dtype=np.float32)

    # Because we undistorted the image, we use the NEW camera matrix and ZERO distortion
    zero_dist = np.zeros((4, 1))
    
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, new_K, zero_dist, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        print("Error: SolvePnP math failed. The points might be physically impossible.")
        return

    # Convert Rotation Vector (rvec) to a 3x3 Rotation Matrix (R)
    R, _ = cv2.Rodrigues(rvec)

    # --- 5. Save Extrinsics AND the active Projection Matrix (P) ---
    # Projection Matrix P = K @ [R | T]
    # We use new_K here because our future pipeline will run on undistorted data
    RT = np.hstack((R, tvec))
    P = new_K @ RT

    extrinsic_data = {
        "rotation_matrix": R.tolist(),
        "translation_vector": tvec.tolist(),
        "projection_matrix": P.tolist(),
        "optimal_camera_matrix": new_K.tolist()
    }

    with open(output_json, 'w') as f:
        json.dump(extrinsic_data, f, indent=4)

    print(f"\nSUCCESS! Camera position calculated.")
    print(f"Height above mat (Z): {-tvec[2][0]:.2f} meters") # -Z is up in OpenCV
    print(f"Saved to {output_json}")

if __name__ == "__main__":
    # Example usage:
    # Get one frame from your synchronized video
    # You run this separately for EVERY camera in your setup
    
    # calibrate_extrinsics_pnp("Screenshot 2026-02-09 at 1.00.58 PM.png", "test.json", "extrinsic_realsense.json", mat_size_meters=1.0)
    # calibrate_extrinsics_pnp("gopro1_frame_001.jpg", "intrinsic_gopro1.json", "extrinsic_gopro1.json", mat_size_meters=8.0)
    pass
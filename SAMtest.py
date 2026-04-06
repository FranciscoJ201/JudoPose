import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import os

# ─────────────────────────────────────────────
#  VISUALIZATION HELPERS
# ─────────────────────────────────────────────

def show_mask(mask, ax, color):
    """
    Overlays a semi-transparent colored mask onto a matplotlib axis.
    
    Arguments:
    - mask (np.ndarray): 2D boolean array representing the segmentation mask.
    - ax (matplotlib.axes.Axes): The matplotlib axis to draw on.
    - color (tuple): RGB color array (e.g., np.array([255/255, 144/255, 30/255])).
    """
    color_img = np.concatenate([color, np.array([0.6])], axis=0) # Add 60% opacity (alpha)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color_img.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    """
    Draws a bounding box outline on a matplotlib axis.
    
    Arguments:
    - box (list or np.ndarray): Bounding box coordinates [x1, y1, x2, y2].
    - ax (matplotlib.axes.Axes): The matplotlib axis to draw on.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# ─────────────────────────────────────────────
#  MAIN TESTING FUNCTION
# ─────────────────────────────────────────────

def test_sam_segmentation(image_path, sam_checkpoint, box_a, box_b, model_type="vit_h"):
    """
    Loads an image, runs SAM using two bounding box prompts, and displays the 
    resulting masks side-by-side for visual inspection.
    
    Arguments:
    - image_path (str): Path to the test image frame (e.g., extracted from your GoPro/RealSense video).
    - sam_checkpoint (str): Path to the downloaded SAM weights (e.g., 'sam_vit_h_4b8939.pth').
    - box_a (list): Bounding box for Athlete A in format [x1, y1, x2, y2].
    - box_b (list): Bounding box for Athlete B in format [x1, y1, x2, y2].
    - model_type (str): The SAM architecture type ('vit_h', 'vit_l', or 'vit_b'). Default is 'vit_h'.
    """
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return
    if not os.path.exists(sam_checkpoint):
        print(f"Error: Could not find SAM checkpoint at {sam_checkpoint}")
        return

    print("Loading image and initializing SAM...")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # SAM expects RGB, not OpenCV's default BGR

    # 1. Initialize the SAM Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 2. Process the Image
    # This computes the image embeddings. It takes a second, but only happens once per image.
    predictor.set_image(image)

    # 3. Predict Mask A
    print("Generating mask for Athlete A...")
    input_box_a = np.array(box_a)
    masks_a, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box_a[None, :],
        multimask_output=False, # We only want the single best mask for the whole person
    )

    # 4. Predict Mask B
    print("Generating mask for Athlete B...")
    input_box_b = np.array(box_b)
    masks_b, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box_b[None, :],
        multimask_output=False,
    )

    # 5. Build the Visual Dashboard
    print("Opening visualizer...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Define distinct colors (Red for A, Blue for B)
    color_a = np.array([255/255, 50/255, 50/255])
    color_b = np.array([50/255, 50/255, 255/255])

    # Left Panel: Athlete A
    axes[0].imshow(image)
    show_mask(masks_a[0], axes[0], color_a)
    show_box(input_box_a, axes[0])
    axes[0].set_title(f"Athlete A Mask\nBox: {box_a}")
    axes[0].axis('off')

    # Right Panel: Athlete B
    axes[1].imshow(image)
    show_mask(masks_b[0], axes[1], color_b)
    show_box(input_box_b, axes[1])
    axes[1].set_title(f"Athlete B Mask\nBox: {box_b}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- Instructions for Testing ---
    # 1. Grab a single frame from your video where the athletes are clinching.
    # 2. Look at your `pipeline_pose.json` output for that exact frame.
    # 3. Copy the two `bbox_xyxy` values and paste them here.
    
    # Example dummy data - replace with your actual JSON outputs
    dummy_image_path = "test_clinch_frame.jpg"
    dummy_sam_weights = "sam_vit_h_4b8939.pth" 
    athlete_a_yolo_box = [150, 50, 400, 800] # [x1, y1, x2, y2]
    athlete_b_yolo_box = [200, 80, 450, 810] # [x1, y1, x2, y2]

    # test_sam_segmentation(
    #     image_path=dummy_image_path,
    #     sam_checkpoint=dummy_sam_weights,
    #     box_a=athlete_a_yolo_box,
    #     box_b=athlete_b_yolo_box
    # )
    pass
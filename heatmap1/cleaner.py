import cv2
import numpy as np
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
IMAGE_PATH = "labeled.png"
OUTPUT_PATH = "labeled_cleaned.png"
TOLERANCE = 40  # Wide range to catch all the gradient/compressed pixels

# Check if the file exists
if not os.path.exists(IMAGE_PATH):
    print(f"Error: {IMAGE_PATH} not found.")
    exit()

# Load the image and convert it to RGB
# cv2.imread() arguments: (filename)
img = cv2.imread(IMAGE_PATH)
# cv2.cvtColor() arguments: (src, code)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create a copy of the image that we will "clean" and overwrite
# np.copy() arguments: (a)
clean_img_rgb = np.copy(img_rgb)

# ==========================================
# 2. EXACT TARGET COLORS
# ==========================================
# List of the exact RGB colors you want to enforce
target_colors = [
    (170, 50, 63),   # Deep Red: Left hand
    (213, 89, 60),   # Orange-Red: Right hand
    (180, 85, 120),  # Pink: Upper chest/shoulders
    (140, 220, 215), # Cyan: Lower abdomen
    (80, 125, 180),  # Blue: Left leg
    (82, 162, 115),  # Green: Right leg
    (243, 239, 153), # Light Yellow: Inner legs
    (139, 102, 60),  # Yellow: Feet
    (176, 242, 77),  # Lime Green: Forearms
    (195, 106, 49),  # Orange: Shoulders / High grip
    (95, 30, 115)    # Deep Circle: Reset point
]

# ==========================================
# 3. COLOR FLATTENING
# ==========================================
print("Cleaning image...")

for target_rgb in target_colors:
    # Calculate bounds with the wide tolerance to catch gradients
    # np.clip() arguments: (a, a_min, a_max)
    lower = np.clip(np.array(target_rgb) - TOLERANCE, 0, 255)
    upper = np.clip(np.array(target_rgb) + TOLERANCE, 0, 255)
    
    # Create the mask for pixels falling within this wider range
    # cv2.inRange() arguments: (src, lowerb, upperb)
    mask = cv2.inRange(img_rgb, lower, upper)
    
    # Overwrite those pixels in our clean image with the EXACT target color
    if np.any(mask):
        clean_img_rgb[mask > 0] = target_rgb
        print(f"Flattened region to pure color: {target_rgb}")

# ==========================================
# 4. SAVE OUTPUT
# ==========================================
# Convert back to BGR for saving, as OpenCV expects BGR format
# cv2.cvtColor() arguments: (src, code)
clean_img_bgr = cv2.cvtColor(clean_img_rgb, cv2.COLOR_RGB2BGR)

# Save the newly flattened image
# cv2.imwrite() arguments: (filename, img)
cv2.imwrite(OUTPUT_PATH, clean_img_bgr)
print(f"\nSuccess! Saved uniform color image as '{OUTPUT_PATH}'")
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

# ==========================================
# 1. CONFIGURATION
# ==========================================
CSV_PATH = "2022_round_2_aga_kagawa_fighter_1.csv"
# The colored image used ONLY for mapping data
LABELED_IMAGE_PATH = "labeled_cleanest.png" 
# The plain white/black outline image used ONLY for the final background
ORIGINAL_IMAGE_PATH = "empty.png" # <-- UPDATE THIS TO YOUR BLANK IMAGE NAME

TOLERANCE = 40  

# Check if files exist
if not os.path.exists(LABELED_IMAGE_PATH):
    print(f"Error: {LABELED_IMAGE_PATH} not found.")
    exit()
if not os.path.exists(ORIGINAL_IMAGE_PATH):
    print(f"Error: {ORIGINAL_IMAGE_PATH} not found.")
    exit()

# Load the CSV data using pandas
# pd.read_csv() arguments: (filepath_or_buffer)
df = pd.read_csv(CSV_PATH)
# df.sum() arguments: (numeric_only)
sums = df.sum(numeric_only=True)

# Load the LABELED image for math
# cv2.imread() arguments: (filename)
img_labeled = cv2.imread(LABELED_IMAGE_PATH)
# cv2.cvtColor() arguments: (src, code)
img_labeled_rgb = cv2.cvtColor(img_labeled, cv2.COLOR_BGR2RGB)
h, w, _ = img_labeled_rgb.shape

# Load the ORIGINAL blank image for the final background
img_original = cv2.imread(ORIGINAL_IMAGE_PATH)
img_original_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

# ==========================================
# 2. COLOR MAPPING (RGB)
# ==========================================
# Keys are your exact RGB values, values are the summed CSV data
color_to_data = {
    (170, 50, 63):   sums.get('left_hand', 0) + (sums.get('Hand_fighting', 0)/2), # Deep Red: Left hand
    (213, 89, 60):   sums.get('right_hand', 0) + (sums.get('Hand_fighting', 0)/2), # Orange-Red: Right hand
    (180, 85, 120):  sums.get('posture_lost', 0) + sums.get('low_collar_grip', 0), # Pink: Upper chest/shoulders
    (140, 220, 215): sums.get('posture_gain', 0),                                # Cyan: Lower abdomen
    (80, 125, 180):  sums.get('left_leg', 0) + sums.get('leg_outside', 0),       # Blue: Left leg
    (82, 162, 115):  sums.get('right_leg', 0) + sums.get('leg_outside', 0),      # Green: Right leg
    (243, 239, 153): sums.get('leg_inside', 0),                                  # Light Yellow: Inner legs
    (139, 102, 60):  sums.get('stepping_foward', 0) + sums.get('stepping_backwards', 0), # Yellow: Feet
    (176, 242, 77):  sums.get('sleeve_grip', 0),                                 # Lime Green: Forearms
    (195, 106, 49):  sums.get('over_hook', 0) + sums.get('high_collar_grip', 0), # Orange: Shoulders / High grip
    (95, 30, 115):   sums.get('reset', 0)                                        # Deep Circle: Reset point
}

# ==========================================
# 3. MASKING & DISTANCE-TRANSFORM GLOW
# ==========================================
# np.zeros() arguments: (shape, dtype)
master_heat = np.zeros((h, w), dtype=np.float32)

# --- NEW: Create a kernel for morphological closing ---
# cv2.getStructuringElement() arguments: (shape, ksize)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

for target_rgb, value in color_to_data.items():
    if value <= 0: continue
    
    # Calculate bounds with the tolerance
    # np.clip() arguments: (a, a_min, a_max) ensures values stay between 0 and 255
    lower = np.clip(np.array(target_rgb) - TOLERANCE, 0, 255)
    upper = np.clip(np.array(target_rgb) + TOLERANCE, 0, 255)
    
    # Create the mask using the LABELED image
    # cv2.inRange() arguments: (src, lowerb, upperb)
    mask = cv2.inRange(img_labeled_rgb, lower, upper)
    
    # --- NEW: Apply morphological closing to fill in "salt and pepper" holes ---
    # cv2.morphologyEx() arguments: (src, op, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    if np.any(mask):
        # Calculate distance from every pixel to the nearest edge
        # cv2.distanceTransform() arguments: (src, distanceType, maskSize)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Find the pixel with the maximum distance (the deepest/thickest part of the shape)
        # cv2.minMaxLoc() arguments: (src)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        
        # max_loc gives us the exact (X, Y) coordinate for our new center
        cX, cY = max_loc
        
        # Create a blank canvas just for this limb
        part_canvas = np.zeros((h, w), dtype=np.float32)
        
        # Place the heat value exactly at the new thickest center point
        part_canvas[cY, cX] = value
        
        # Apply Gaussian blur to create the radiating glow
        # gaussian_filter() arguments: (input, sigma)
        blurred_part = gaussian_filter(part_canvas, sigma=70)
        
        # Scale the blurred heat back up so it stays visible after diffusing
        if blurred_part.max() > 0:
            blurred_part = (blurred_part / blurred_part.max()) * value
        
        # THE COOKIE CUTTER: Only keep the glow that falls INSIDE the fixed, closed mask
        # np.where() arguments: (condition, x, y)
        cookie_cut_heat = np.where(mask > 0, blurred_part, 0)
        
        # Add this perfectly cut limb to the master heatmap layer
        master_heat += cookie_cut_heat

# ==========================================
# 4. PLOTTING
# ==========================================
if master_heat.max() > 0:
    master_heat_norm = master_heat / master_heat.max()
else:
    master_heat_norm = master_heat

# Apply the 'turbo' colormap (Blue -> Green -> Yellow -> Red)
# plt.get_cmap() arguments: (name)
cmap = plt.get_cmap('turbo')
colored_heat = cmap(master_heat_norm)

# Set transparency: areas with 0 data are hidden, others are 85% visible
# np.where() arguments: (condition, x, y)
colored_heat[..., 3] = np.where(master_heat > 0, 0.85, 0)

# plt.subplots() arguments: (figsize)
fig, ax = plt.subplots(figsize=(10, 12))

# Plot the ORIGINAL blank background first
# ax.imshow() arguments: (X, cmap)
ax.imshow(img_original_gray, cmap='gray')

# Overlay the data-driven heatmap
ax.imshow(colored_heat)
ax.axis('off')

# plt.savefig() arguments: (fname, bbox_inches, dpi)
plt.savefig("judo_heatmap_result.png", bbox_inches='tight', dpi=300)
print("Saved dual-image result with morphological closing to 'judo_heatmap_result.png'")

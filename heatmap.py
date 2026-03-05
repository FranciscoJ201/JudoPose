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
IMAGE_PATH = "labeled.png"

# --- INCREASED TOLERANCE ---
# Bumped up to 40 to catch the entire color gradient inside the shapes
TOLERANCE = 60  

if not os.path.exists(IMAGE_PATH):
    print(f"Error: {IMAGE_PATH} not found.")
    exit()

# Load the CSV data using pandas
# pd.read_csv() arguments: (filepath_or_buffer)
df = pd.read_csv(CSV_PATH)
# df.sum() arguments: (numeric_only)
sums = df.sum(numeric_only=True)

# Load the image and convert it
# cv2.imread() arguments: (filename)
img = cv2.imread(IMAGE_PATH)
# cv2.cvtColor() arguments: (src, code)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape

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
# 3. MASKING & COOKIE-CUTTER GLOW (High Tolerance)
# ==========================================
# np.zeros() arguments: (shape, dtype)
master_heat = np.zeros((h, w), dtype=np.float32)

for target_rgb, value in color_to_data.items():
    if value <= 0: continue
    
    # Calculate bounds with the increased tolerance
    # np.clip() arguments: (a, a_min, a_max) ensures values stay between 0 and 255
    lower = np.clip(np.array(target_rgb) - TOLERANCE, 0, 255)
    upper = np.clip(np.array(target_rgb) + TOLERANCE, 0, 255)
    
    # Create the mask for pixels falling within this wider range
    # cv2.inRange() arguments: (src, lowerb, upperb)
    mask = cv2.inRange(img_rgb, lower, upper)
    
    if np.any(mask):
        # Find the center point of this specific body part using image moments
        # cv2.moments() arguments: (array)
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Create a blank canvas just for this limb
            part_canvas = np.zeros((h, w), dtype=np.float32)
            
            # Place the heat value exactly at the center point
            part_canvas[cY, cX] = value
            
            # Apply Gaussian blur to create the radiating glow
            # gaussian_filter() arguments: (input, sigma)
            blurred_part = gaussian_filter(part_canvas, sigma=70)
            
            # Scale the blurred heat back up so it stays visible after diffusing
            if blurred_part.max() > 0:
                blurred_part = (blurred_part / blurred_part.max()) * value
            
            # THE COOKIE CUTTER: Only keep the glow that falls INSIDE the wide mask
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

# Convert original image to grayscale so the colors pop
# cv2.cvtColor() arguments: (src, code)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# plt.subplots() arguments: (figsize)
fig, ax = plt.subplots(figsize=(10, 12))

# Plot the grayscale background first
# ax.imshow() arguments: (X, cmap)
ax.imshow(img_gray, cmap='gray')

# Overlay the data-driven heatmap
ax.imshow(colored_heat)
ax.axis('off')

# plt.savefig() arguments: (fname, bbox_inches, dpi)
plt.savefig("judo_heatmap_result.png", bbox_inches='tight', dpi=300)
print("Saved high-tolerance cookie-cutter result to 'judo_heatmap_result.png'")
plt.show()
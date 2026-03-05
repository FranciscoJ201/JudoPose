import cv2
import numpy as np

# Load your image
img = cv2.imread("labeled.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape to a list of pixels
pixels = img_rgb.reshape(-1, 3)

# Filter out the whites (background) and blacks (lines)
# We only want pixels where at least one color channel is NOT near 255 or 0
mask = np.all((pixels < 240) & (pixels > 15), axis=1)
colored_pixels = pixels[mask]

if len(colored_pixels) == 0:
    print("No colored regions found! Check if you used a 'Pencil' tool or if the image is just black and white.")
else:
    unique_colors, counts = np.unique(colored_pixels, axis=0, return_counts=True)
    sorted_indices = np.argsort(-counts)
    
    print("Detected Region Colors (R, G, B):")
    for i in range(min(15, len(unique_colors))):
        idx = sorted_indices[i]
        print(f"Color: {unique_colors[idx]} - Pixel Count: {counts[idx]}")
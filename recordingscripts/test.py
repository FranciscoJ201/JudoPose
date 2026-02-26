import torch
import time

print("--- GPU Math Test ---")
try:
    # 1. Create a matrix on the GPU
    x = torch.rand(1000, 1000).to("cuda")
    y = torch.rand(1000, 1000).to("cuda")
    
    # 2. Perform a heavy matrix multiplication
    start = time.time()
    z = x @ y
    end = time.time()
    
    print(f"Success! Matrix multiplication result shape: {z.shape}")
    print(f"Time taken: {end - start:.4f} seconds")
    print("You can ignore the compatibility warning.")
    
except Exception as e:
    print("\nCRITICAL FAILURE:")
    print(e)
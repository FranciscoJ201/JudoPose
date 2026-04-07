import os
import sys
import torch
import numpy as np
import cv2
import torch.nn.functional as F

# Add the XMem folder to the Python path so we can import its modules
sys.path.append(os.path.abspath('XMem'))

from model.network import XMem
from inference.inference_core import InferenceCore
from dataset.range_transform import im_normalization

class XMemBridge:
    def __init__(self, checkpoint_path='XMem/XMem.pth', device='cuda'):
        """
        Initializes the XMem Video Object Segmentation model and its memory core.
        """
        self.device = device
        print(f"Loading XMem weights from {checkpoint_path}...")
        
        # 1. Load the neural network
        self.network = XMem({}, 'resnet50').to(self.device).eval()
        model_weights = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(model_weights)
        
        # 2. Initialize the Memory Core
        # XMem tracks multiple objects via a central memory bank
        self.processor = InferenceCore(self.network, config={})
        self.is_initialized = False
        self.num_objects = 2 # Athlete A and Athlete B

    def _prepare_image(self, frame_bgr):
        """Converts OpenCV BGR image to the normalized Tensor XMem expects."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        tensor = im_normalization(tensor).to(self.device)
        return tensor.unsqueeze(0) # Add batch dimension

    def initialize_masks(self, first_frame_bgr, mask_a, mask_b):
        """
        Feeds the very first frame and the SAM-generated masks into XMem to establish its memory.
        
        Arguments:
        - first_frame_bgr: The raw OpenCV image of the first clinch frame.
        - mask_a: 2D numpy boolean array for Athlete A (from SAM).
        - mask_b: 2D numpy boolean array for Athlete B (from SAM).
        """
        image_tensor = self._prepare_image(first_frame_bgr)
        
        # Create a single multi-class mask: 0=Background, 1=Athlete A, 2=Athlete B
        combined_mask = np.zeros(first_frame_bgr.shape[:2], dtype=np.uint8)
        combined_mask[mask_a == True] = 1
        combined_mask[mask_b == True] = 2
        
        # Convert to one-hot tensor
        mask_tensor = torch.from_numpy(combined_mask).long().to(self.device)
        mask_onehot = F.one_hot(mask_tensor, num_classes=self.num_objects + 1)
        mask_onehot = mask_onehot.permute(2, 0, 1).float().unsqueeze(0) # [B, C, H, W]
        
        # Feed the first frame to the processor
        self.processor.set_all_labels(list(range(1, self.num_objects + 1)))
        self.processor.step(image_tensor, mask_onehot[0]) 
        
        self.is_initialized = True
        print("XMem Memory initialized with SAM masks.")

    def track_frame(self, frame_bgr):
        """
        Passes a new frame into XMem. XMem compares it to its memory bank 
        and outputs the new locations of Athlete A and B.
        
        Returns:
        - mask_a, mask_b (as 2D boolean numpy arrays)
        """
        if not self.is_initialized:
            raise RuntimeError("You must call initialize_masks() before track_frame().")
            
        image_tensor = self._prepare_image(frame_bgr)
        
        # Step the processor forward (no mask provided, XMem predicts it)
        prediction = self.processor.step(image_tensor) 
        
        # prediction is shape [C, H, W], get the argmax to find the winning class per pixel
        pred_classes = torch.argmax(prediction, dim=0).cpu().numpy().astype(np.uint8)
        
        # Split back into individual boolean masks
        mask_a = (pred_classes == 1)
        mask_b = (pred_classes == 2)
        
        return mask_a, mask_b
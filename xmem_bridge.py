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
        
        # 1. Fully Unified configuration for XMem
        # Renamed 'enable_long_term_usage' to 'enable_long_term_count_usage' 
        # and added architectural constants.
        self.config = {
            'top_k': 30,
            'mem_every': 5,
            'deep_update_every': -1,
            'enable_long_term': True,
            'enable_long_term_count_usage': True, # FIXED: Matched exact key name
            'max_mid_term_frames': 10,
            'min_mid_term_frames': 5,
            'max_long_term_nodes': 10000,
            'max_long_term_elements': 10000,
            'key_dim': 64,
            'value_dim': 512,
            'hidden_dim': 64,
            'num_prototypes': 128,
            'stochastic_sampling': False,          # Safety key
            'sam_threshold': 0.1                   # Safety key
        }
        
        # 2. Initialize network
        self.network = XMem(config=self.config, model_path=None).to(self.device).eval()
        model_weights = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(model_weights)
        
        # 3. Initialize the Memory Core
        self.processor = InferenceCore(self.network, config=self.config)
        self.is_initialized = False
        self.num_objects = 2 # Athlete A and Athlete B

    def _prepare_image(self, frame_bgr):
        # RESIZE HERE: Work at 480p internal resolution to save massive VRAM
        # XMem tracks better at lower resolutions for fast sports movement anyway
        self.orig_h, self.orig_w = frame_bgr.shape[:2]
        frame_resized = cv2.resize(frame_bgr, (854, 480)) 
        
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        tensor = im_normalization(tensor).to(self.device)
        return tensor

    def initialize_masks(self, first_frame_bgr, mask_a, mask_b):
        """
        Feeds the first frame and ONLY the 2 foreground masks into XMem.
        """
        # 1. This resizes the image to 480p internally
        image_tensor = self._prepare_image(first_frame_bgr)
        
        # 2. RESIZE THE MASKS to match the 480p proxy (854x480)
        # We use INTER_NEAREST because these are binary masks; we don't want fuzzy edges.
        mask_a_resized = cv2.resize(mask_a.astype(np.uint8), (854, 480), interpolation=cv2.INTER_NEAREST)
        mask_b_resized = cv2.resize(mask_b.astype(np.uint8), (854, 480), interpolation=cv2.INTER_NEAREST)
        
        # 3. Stack the resized masks
        masks = np.stack([mask_a_resized, mask_b_resized], axis=0).astype(np.float32) 
        mask_tensor = torch.from_numpy(masks).to(self.device) 
        
        # 4. Initialize the processor
        self.processor.set_all_labels([1, 2])
        self.processor.step(image_tensor, mask_tensor) 
        
        self.is_initialized = True
        print(f"XMem Memory initialized for {self.num_objects} athletes at 480p.")

    def track_frame(self, frame_bgr):
        image_tensor = self._prepare_image(frame_bgr)
        prediction = self.processor.step(image_tensor) 
        
        # Scale the prediction back up to match the original video size
        prediction = F.interpolate(prediction.unsqueeze(0), 
                                   size=(self.orig_h, self.orig_w), 
                                   mode='bilinear', align_corners=False)[0]
        
        pred_classes = torch.argmax(prediction, dim=0).cpu().numpy().astype(np.uint8)
        return (pred_classes == 1), (pred_classes == 2)
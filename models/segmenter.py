"""
Segmenter Module - MedSAM with YOLO Fallback
Fully CPU compatible - Fixed version
"""

import os
import torch
import numpy as np
from PIL import Image
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MedSAMDetector:
    """
    MedSAM-based lesion detector with YOLO fallback
    Fully CPU compatible
    """
    
    def __init__(self, checkpoint_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fallback = False
        self.yolo_model = None
        self.sam = None
        self.mask_generator = None
        
        # Try to load MedSAM
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                
                print(f"📥 Loading MedSAM from: {checkpoint_path}")
                
                # ============================================================
                # FIX 1: Load checkpoint with CPU mapping
                # ============================================================
                # This is the key fix - load with map_location='cpu'
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                
                # ============================================================
                # FIX 2: Create model with custom loading
                # ============================================================
                # Instead of using the default loading, we manually load the state dict
                
                # First, create the model architecture
                from segment_anything import sam_model_registry
                
                # Create model with the correct architecture
                # We use a dummy checkpoint path to create the model
                self.sam = sam_model_registry["vit_b"](checkpoint=None)
                
                # Load the state dict from the checkpoint
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                
                # Remove 'model.' prefix if it exists (for compatibility)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                
                # Load state dict
                self.sam.load_state_dict(new_state_dict)
                self.sam.to(device=self.device)
                self.sam.eval()
                
                # Create mask generator
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=self.sam,
                    points_per_side=8,
                    pred_iou_thresh=0.5,
                    stability_score_thresh=0.5,
                    crop_n_layers=0,
                    min_mask_region_area=100,
                )
                print(f"✅ MedSAM loaded successfully on {self.device}")
                return
                
            except Exception as e:
                print(f"⚠️ MedSAM loading failed: {e}")
                import traceback
                traceback.print_exc()
        
        # If we get here, MedSAM failed - use YOLO fallback
        self.use_fallback = True
        self._init_yolo()
    
    def _init_yolo(self):
        """Initialize YOLO model"""
        try:
            from ultralytics import YOLO
            
            # Check for best.pt in common locations
            yolo_paths = [
                "best.pt",
                "./best.pt",
                os.path.join(os.getcwd(), "best.pt"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best.pt"),
                "/home/koosha/pp/RMS-oral/best.pt",
            ]
            
            for path in yolo_paths:
                if os.path.exists(path):
                    self.yolo_model = YOLO(path)
                    self.yolo_model.to('cpu')
                    print(f"✅ YOLO fallback ready ({path})")
                    return
            
            print("⚠️ YOLO model not found. Only semantic analysis will work.")
            self.yolo_model = None
        except Exception as e:
            print(f"⚠️ YOLO init failed: {e}")
            self.yolo_model = None
    
    def detect_lesions(self, image):
        """Detect lesions using MedSAM or YOLO fallback"""
        # If MedSAM is available, try it first
        if not self.use_fallback and self.sam is not None and self.mask_generator is not None:
            try:
                # Resize for performance
                if image.size[0] > 800 or image.size[1] > 800:
                    image.thumbnail((800, 800), Image.Resampling.LANCZOS)
                
                image_np = np.array(image)
                masks = self.mask_generator.generate(image_np)
                
                if masks:
                    filtered = []
                    for mask in masks:
                        if mask['area'] > 1000:
                            filtered.append({
                                'segmentation': mask['segmentation'],
                                'bbox': [int(x) for x in mask['bbox']],
                                'area': int(mask['area']),
                                'confidence': float(mask['predicted_iou']),
                                'crop': self._extract_crop(image, mask['bbox'])
                            })
                    
                    filtered.sort(key=lambda x: x['confidence'], reverse=True)
                    print(f"✅ MedSAM detected {len(filtered)} lesions")
                    return filtered
                else:
                    print("⚠️ MedSAM found no masks, using YOLO")
            except Exception as e:
                print(f"⚠️ MedSAM error: {e}")
        
        # Fallback to YOLO
        return self._yolo_fallback(image)
    
    def _yolo_fallback(self, image):
        """YOLO fallback detection"""
        if self.yolo_model is None:
            return []
        
        try:
            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            results = self.yolo_model(image, conf=0.25)[0]
            if results.boxes is None:
                return []
            
            boxes = results.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            
            output = []
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                output.append({
                    'segmentation': None,
                    'bbox': [x1, y1, x2, y2],
                    'area': (x2-x1) * (y2-y1),
                    'confidence': float(confs[i]),
                    'crop': image.crop((x1, y1, x2, y2))
                })
            
            if output:
                print(f"✅ YOLO detected {len(output)} lesions")
            return output
        except Exception as e:
            print(f"⚠️ YOLO error: {e}")
            return []
    
    def _extract_crop(self, image, bbox):
        """Extract crop from bbox"""
        try:
            x, y, w, h = bbox
            return image.crop((x, y, x+w, y+h))
        except:
            return None
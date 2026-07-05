"""
Safety Checker - OOD Detection
"""

import numpy as np
from PIL import Image
from typing import Dict, Any
import cv2
from config import Config


class SafetyChecker:
    """Safety checker for input validation"""
    
    def __init__(self):
        self.min_resolution = getattr(Config, 'MIN_RESOLUTION', (200, 200))
        self.max_resolution = getattr(Config, 'MAX_RESOLUTION', (4000, 4000))
        self.min_brightness = getattr(Config, 'MIN_BRIGHTNESS', 30)
        self.max_brightness = getattr(Config, 'MAX_BRIGHTNESS', 230)
        self.min_contrast = getattr(Config, 'MIN_CONTRAST', 20)
        self.min_blur_threshold = getattr(Config, 'MIN_BLUR_THRESHOLD', 100)
    
    def check(self, image) -> Dict[str, Any]:
        """Check image validity"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        if not isinstance(img_array, np.ndarray):
            return {'passed': False, 'reason': 'Invalid image format', 'metrics': {}}
        
        metrics = {}
        
        try:
            h, w = img_array.shape[:2]
            metrics['width'] = int(w)
            metrics['height'] = int(h)
            
            if w < self.min_resolution[0] or h < self.min_resolution[1]:
                return {'passed': False, 'reason': f'Resolution too low: {w}x{h}', 'metrics': metrics}
            
            if w > self.max_resolution[0] or h > self.max_resolution[1]:
                return {'passed': False, 'reason': f'Resolution too high: {w}x{h}', 'metrics': metrics}
            
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            brightness = float(np.mean(gray))
            metrics['brightness'] = brightness
            if brightness < self.min_brightness or brightness > self.max_brightness:
                return {'passed': False, 'reason': f'Brightness issue: {brightness:.1f}', 'metrics': metrics}
            
            contrast = float(np.std(gray))
            metrics['contrast'] = contrast
            if contrast < self.min_contrast:
                return {'passed': False, 'reason': f'Low contrast: {contrast:.1f}', 'metrics': metrics}
            
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            metrics['blur_score'] = laplacian_var
            if laplacian_var < self.min_blur_threshold:
                return {'passed': False, 'reason': f'Too blurry: {laplacian_var:.1f}', 'metrics': metrics}
            
            aspect_ratio = float(w / h)
            metrics['aspect_ratio'] = aspect_ratio
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                return {'passed': False, 'reason': f'Unusual aspect ratio: {aspect_ratio:.2f}', 'metrics': metrics}
            
            return {
                'passed': True,
                'reason': 'Image passed safety check',
                'metrics': metrics
            }
            
        except Exception as e:
            return {'passed': False, 'reason': f'Safety check error: {str(e)}', 'metrics': metrics}

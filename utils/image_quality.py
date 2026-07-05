"""
Image Quality Assessment
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any
from config import Config


class ImageQualityAssessor:
    """Assess image quality"""
    
    def __init__(self):
        self.min_resolution = getattr(Config, 'MIN_RESOLUTION', (200, 200))
        self.max_resolution = getattr(Config, 'MAX_RESOLUTION', (4000, 4000))
        self.min_brightness = getattr(Config, 'MIN_BRIGHTNESS', 30)
        self.max_brightness = getattr(Config, 'MAX_BRIGHTNESS', 230)
        self.min_contrast = getattr(Config, 'MIN_CONTRAST', 20)
        self.min_blur_threshold = getattr(Config, 'MIN_BLUR_THRESHOLD', 100)
    
    def assess(self, image) -> Dict[str, Any]:
        """Assess image quality"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        results = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        try:
            h, w = img_array.shape[:2]
            results['metrics']['width'] = int(w)
            results['metrics']['height'] = int(h)
            
            if w < self.min_resolution[0] or h < self.min_resolution[1]:
                results['passed'] = False
                results['issues'].append(f"Resolution too low: {w}x{h}")
            
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            brightness = float(np.mean(gray))
            results['metrics']['brightness'] = brightness
            if brightness < self.min_brightness or brightness > self.max_brightness:
                results['passed'] = False
                results['issues'].append(f"Brightness: {brightness:.1f}")
            
            contrast = float(np.std(gray))
            results['metrics']['contrast'] = contrast
            if contrast < self.min_contrast:
                results['passed'] = False
                results['issues'].append(f"Low contrast: {contrast:.1f}")
            
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            results['metrics']['blur_score'] = laplacian_var
            if laplacian_var < self.min_blur_threshold:
                results['passed'] = False
                results['issues'].append(f"Too blurry: {laplacian_var:.1f}")
            
            aspect_ratio = float(w / h)
            results['metrics']['aspect_ratio'] = aspect_ratio
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                results['passed'] = False
                results['issues'].append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
            
        except Exception as e:
            results['passed'] = False
            results['issues'].append(f"Assessment error: {str(e)}")
        
        return results

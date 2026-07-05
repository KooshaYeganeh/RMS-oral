"""
Feature Extraction - Geometric and Clinical Features
"""

import numpy as np
import cv2
from typing import Dict, Any, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MaskFeatureExtractor:
    """Extract geometric and color features from segmentation masks"""
    
    def extract(self, mask: np.ndarray, image: np.ndarray = None) -> Dict[str, Any]:
        """Extract features from segmentation mask"""
        features = {}
        
        if mask is None:
            return features
        
        try:
            # Convert to uint8
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return features
            
            contour = max(contours, key=cv2.contourArea)
            
            # Geometric features
            area = cv2.contourArea(contour)
            features['area'] = float(area)
            
            perimeter = cv2.arcLength(contour, True)
            features['perimeter'] = float(perimeter)
            
            if perimeter > 0:
                features['circularity'] = float(4 * np.pi * area / (perimeter ** 2))
            else:
                features['circularity'] = 1.0
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                features['solidity'] = float(area / hull_area)
            else:
                features['solidity'] = 1.0
            
            x, y, w, h = cv2.boundingRect(contour)
            if h > 0:
                features['aspect_ratio'] = float(w / h)
            else:
                features['aspect_ratio'] = 1.0
            
            if w * h > 0:
                features['extent'] = float(area / (w * h))
            else:
                features['extent'] = 0.0
            
            # Color features
            if image is not None and len(image.shape) == 3:
                region_mask = mask > 0
                total_pixels = np.sum(region_mask)
                
                if total_pixels > 0:
                    r = image[:, :, 0][region_mask]
                    g = image[:, :, 1][region_mask]
                    b = image[:, :, 2][region_mask]
                    
                    features['mean_r'] = float(np.mean(r))
                    features['mean_g'] = float(np.mean(g))
                    features['mean_b'] = float(np.mean(b))
                    
                    features['var_r'] = float(np.var(r))
                    features['var_g'] = float(np.var(g))
                    features['var_b'] = float(np.var(b))
                    
                    white_pixels = np.sum((r > 200) & (g > 200) & (b > 200))
                    features['white_ratio'] = float(white_pixels / total_pixels)
                    
                    red_pixels = np.sum(r > 150)
                    features['red_ratio'] = float(red_pixels / total_pixels)
                    
                    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    hsv_region = hsv[region_mask]
                    
                    features['mean_hue'] = float(np.mean(hsv_region[:, 0]))
                    features['mean_sat'] = float(np.mean(hsv_region[:, 1]))
                    features['mean_val'] = float(np.mean(hsv_region[:, 2]))
                    
                    features['var_hue'] = float(np.var(hsv_region[:, 0]))
                    features['var_sat'] = float(np.var(hsv_region[:, 1]))
                    features['var_val'] = float(np.var(hsv_region[:, 2]))
            
            return features
            
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
            return features


class ClinicalFeatureExtractor:
    """Extract clinical features from aggregated data"""
    
    def extract(self, features: Dict, semantic: Dict, lesions: List) -> Dict:
        """Extract clinical features"""
        clinical = {
            'lesion_count': features.get('lesion_count', 0),
            'area': features.get('area', 0),
            'circularity': features.get('circularity', 1.0),
            'solidity': features.get('solidity', 1.0),
            'white_ratio': features.get('white_ratio', 0),
            'red_ratio': features.get('red_ratio', 0),
            'aspect_ratio': features.get('aspect_ratio', 1.0),
            'extent': features.get('extent', 0),
            'hue_var': features.get('var_hue', 0),
            'sat_var': features.get('var_sat', 0),
            'val_var': features.get('var_val', 0),
        }
        
        # Binary features
        clinical['irregular_border'] = clinical['circularity'] < 0.5
        clinical['white_plaque'] = clinical['white_ratio'] > 0.2
        clinical['large_lesion'] = clinical['area'] > 80
        clinical['multiple_lesions'] = clinical['lesion_count'] > 1
        
        # Semantic features
        if semantic and 'ensemble' in semantic:
            label = semantic['ensemble'].get('label', '')
            clinical['semantic_label'] = label
            clinical['semantic_confidence'] = semantic['ensemble'].get('confidence', 0.5)
            clinical['semantic_uncertainty'] = semantic['ensemble'].get('uncertainty', 0.3)
            
            # Check if abnormal
            abnormal_labels = ['dental caries', 'gingivitis', 'periodontitis', 'oral lesion', 'oral cancer suspicion']
            clinical['is_abnormal'] = label in abnormal_labels
        
        # Clinical terms
        terms = []
        if clinical['irregular_border']:
            terms.append('irregular_border')
        if clinical['white_plaque']:
            terms.append('white_plaque')
        if clinical.get('is_abnormal', False):
            terms.append(clinical.get('semantic_label', ''))
        if clinical['multiple_lesions']:
            terms.append('multiple_lesions')
        if clinical['large_lesion']:
            terms.append('large_lesion')
        
        clinical['terms'] = list(set([t for t in terms if t]))
        
        # Risk factors
        clinical['bleeding'] = False  # Placeholder
        clinical['ulcer'] = 'ulcer' in clinical['terms']
        clinical['swelling'] = clinical['area'] > 100
        clinical['erythema'] = clinical['red_ratio'] > 0.3
        clinical['location'] = 'unknown'  # Placeholder
        
        return clinical
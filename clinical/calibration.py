"""
Model Calibration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from config import Config


class ModelCalibrator:
    """Model calibration using temperature scaling"""
    
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))
        self.is_calibrated = False
        self.calibration_history = []
    
    def calibrate_predictions(self, semantic: Dict, features: Dict, clinical_features: Dict) -> Dict:
        """Calibrate predictions"""
        result = {
            'is_calibrated': self.is_calibrated,
            'temperature': float(self.temperature.item()) if self.is_calibrated else 1.0,
            'uncertainty': 0.0
        }
        
        if not semantic or 'ensemble' not in semantic:
            result['uncertainty'] = 0.5
            return result
        
        uncertainty = semantic['ensemble'].get('uncertainty', 0.3)
        
        if self.is_calibrated:
            scaled_uncertainty = uncertainty * float(self.temperature.item())
            result['uncertainty'] = min(1.0, scaled_uncertainty)
        else:
            result['uncertainty'] = uncertainty
        
        # Additional calibration
        if clinical_features.get('lesion_count', 0) > 0 and semantic['ensemble'].get('confidence', 0.5) < 0.5:
            result['uncertainty'] = min(1.0, result['uncertainty'] + 0.2)
        
        if clinical_features.get('lesion_count', 0) > 2:
            result['uncertainty'] = max(0.1, result['uncertainty'] - 0.1)
        
        return result

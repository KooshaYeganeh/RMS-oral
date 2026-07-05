"""
Feature-Based Risk Engine
"""

from typing import Dict, Any, Optional
import numpy as np
from config import Config


class RiskEngine:
    """Clinical risk engine based on features"""
    
    def __init__(self):
        self.weights = getattr(Config, 'FEATURE_WEIGHTS', {
            'lesion_presence': 0.30,
            'irregular_border': 0.25,
            'white_plaque': 0.20,
            'ulceration': 0.15,
            'size_large': 0.10,
            'high_risk_location': 0.10
        })
        self.high_risk_locations = getattr(Config, 'HIGH_RISK_LOCATIONS', [])
        self.components = {}
    
    def calculate(self, aggregated_features, clinical_features, semantic, override, calibrated):
        """Calculate risk score"""
        self.components = {}
        risk_score = 0.0
        
        # 1. Lesion presence
        lesion_count = aggregated_features.get('lesion_count', 0)
        if lesion_count > 0:
            lesion_score = min(1.0, lesion_count * 0.25)
            self.components['lesion_presence'] = lesion_score * self.weights['lesion_presence']
            risk_score += self.components['lesion_presence']
        
        # 2. Irregular border
        circularity = aggregated_features.get('circularity', 1.0)
        if circularity < 0.5:
            irregular_score = 1.0
        elif circularity < 0.7:
            irregular_score = 0.5
        else:
            irregular_score = 0.0
        self.components['irregular_border'] = irregular_score * self.weights['irregular_border']
        risk_score += self.components['irregular_border']
        
        # 3. White plaque
        white_ratio = aggregated_features.get('white_ratio', 0)
        if white_ratio > 0.3:
            white_score = 1.0
        elif white_ratio > 0.15:
            white_score = 0.5
        else:
            white_score = 0.0
        self.components['white_plaque'] = white_score * self.weights['white_plaque']
        risk_score += self.components['white_plaque']
        
        # 4. Ulceration
        if clinical_features.get('ulcer', False):
            self.components['ulceration'] = self.weights['ulceration']
            risk_score += self.components['ulceration']
        
        # 5. Size
        area = aggregated_features.get('area', 0)
        if area > 100:
            size_score = 1.0
        elif area > 50:
            size_score = 0.5
        else:
            size_score = 0.0
        self.components['size_large'] = size_score * self.weights['size_large']
        risk_score += self.components['size_large']
        
        # 6. High risk location
        location = clinical_features.get('location', '')
        if location in self.high_risk_locations:
            self.components['high_risk_location'] = self.weights['high_risk_location']
            risk_score += self.components['high_risk_location']
        
        # 7. Clinical override
        if override.get('applied'):
            override_score = override.get('override_risk', 0.5)
            risk_score = max(risk_score, override_score)
            self.components['clinical_override'] = override_score
        
        return min(1.0, max(0.0, risk_score))
    
    def get_components(self):
        return self.components.copy()
    
    def get_level(self, risk_score):
        if risk_score >= getattr(Config, 'RISK_THRESHOLD_HIGH', 0.7):
            return 'high'
        elif risk_score >= getattr(Config, 'RISK_THRESHOLD_MEDIUM', 0.4):
            return 'medium'
        else:
            return 'low'
    
    def get_message(self, risk_score):
        level = self.get_level(risk_score)
        messages = {
            'high': '🔴 High risk - Immediate specialist referral needed',
            'medium': '🟡 Moderate risk - Schedule evaluation within 1 month',
            'low': '🟢 Low risk - Routine dental check-up'
        }
        return messages.get(level, 'Risk assessment completed')

"""
RMS-ORAL Configuration
"""

import os
import torch


class Config:
    """Global configuration"""
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = "/tmp/uploads"
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model paths
    VIT_MODEL_PATH = os.path.join(MODELS_DIR, "vit_oral_finetune_model")
    MEDSAM_CHECKPOINT = os.path.join(MODELS_DIR, "medsam", "medsam_vit_b.pth")
    YOLO_MODEL_PATH = "best.pt"
    
    # Image Quality Thresholds
    MIN_RESOLUTION = (200, 200)
    MAX_RESOLUTION = (4000, 4000)
    MIN_BRIGHTNESS = 30
    MAX_BRIGHTNESS = 230
    MIN_CONTRAST = 20
    MIN_BLUR_THRESHOLD = 100
    
    # Clinical Thresholds
    RISK_THRESHOLD_HIGH = 0.7
    RISK_THRESHOLD_MEDIUM = 0.4
    UNCERTAINTY_THRESHOLD = 0.5
    
    # Feature Weights
    FEATURE_WEIGHTS = {
        'lesion_presence': 0.30,
        'irregular_border': 0.25,
        'white_plaque': 0.20,
        'ulceration': 0.15,
        'size_large': 0.10,
        'high_risk_location': 0.10,
        'texture_abnormal': 0.10,
        'color_variance': 0.05,
    }
    
    # High Risk Locations
    HIGH_RISK_LOCATIONS = ['tongue_bottom', 'floor_mouth', 'soft_palate', 'lateral_tongue']
    
    # Abnormal Labels
    ABNORMAL_LABELS = [
        'dental caries', 'gingivitis', 'periodontitis',
        'oral lesion', 'oral cancer suspicion', 'leukoplakia',
        'erythroplakia', 'ulcer'
    ]
    
    # Clinical Override Rules
    CLINICAL_OVERRIDE_RULES = {
        'lesion_with_healthy': {
            'condition': lambda x: x.get('lesion_count', 0) > 0 and x.get('semantic_label') == 'healthy mouth',
            'override_risk': 0.5,
            'message': 'Lesion detected but model indicates healthy - Clinical override applied'
        },
        'multiple_irregular': {
            'condition': lambda x: x.get('irregular_count', 0) >= 2,
            'override_risk': 0.7,
            'message': 'Multiple irregular lesions detected - High risk override'
        }
    }
    
    # Tracking
    MAX_TRACKING_HISTORY = 10
    TRACKING_CHANGE_THRESHOLD = 0.2
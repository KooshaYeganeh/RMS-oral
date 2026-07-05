"""
Clinical Decision Engine - Complete Clinical Decision System
"""
import os
import sys
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

class ClinicalDecisionEngine:
    """Complete Clinical Decision Engine"""
    
    def __init__(self):
        print("=" * 60)
        print("🩺 Initializing Clinical Decision Engine")
        print("=" * 60)
        
        # Import components inside __init__ to avoid circular imports
        self._init_components()
        
        # Tracking
        self.tracking_history = {}
        
        print("✅ Clinical Decision Engine initialized")
        print("=" * 60)
    
    def _init_components(self):
        """Initialize all components with proper imports"""
        
        # ========== Safety & Quality ==========
        try:
            from utils.image_quality import ImageQualityAssessor
            from clinical.safety import SafetyChecker
            
            self.quality_assessor = ImageQualityAssessor()
            self.safety_checker = SafetyChecker()
            print("✅ Quality & Safety components loaded")
        except Exception as e:
            print(f"⚠️ Quality/Safety components failed: {e}")
            self.quality_assessor = None
            self.safety_checker = None
        
        # ========== Core Models ==========
        try:
            from models.segmenter import MedSAMDetector
            from models.semantic import SemanticEnsemble
            
            self.segmenter = MedSAMDetector(checkpoint_path=Config.MEDSAM_CHECKPOINT)
            self.semantic = SemanticEnsemble(vit_model_path=Config.VIT_MODEL_PATH)
            self.semantic.load_biomedclip()
            self.semantic.load_clinical_vit()
            print("✅ Core models loaded")
        except Exception as e:
            print(f"⚠️ Core models failed: {e}")
            self.segmenter = None
            self.semantic = None
        
        # ========== Feature Extractors ==========
        try:
            from models.features import MaskFeatureExtractor, ClinicalFeatureExtractor
            
            self.mask_feature_extractor = MaskFeatureExtractor()
            self.clinical_feature_extractor = ClinicalFeatureExtractor()
            print("✅ Feature extractors loaded")
        except Exception as e:
            print(f"⚠️ Feature extractors failed: {e}")
            self.mask_feature_extractor = None
            self.clinical_feature_extractor = None
        
        # ========== Clinical Components ==========
        try:
            from clinical.risk_engine import RiskEngine
            from clinical.differential import DifferentialDiagnosis
            from clinical.guidelines import GuidelineEngine
            from clinical.calibration import ModelCalibrator
            from utils.report import ReportGenerator
            
            self.risk_engine = RiskEngine()
            self.differential = DifferentialDiagnosis()
            self.guidelines = GuidelineEngine()
            self.calibrator = ModelCalibrator()
            self.report_generator = ReportGenerator()
            print("✅ Clinical components loaded")
        except Exception as e:
            print(f"⚠️ Clinical components failed: {e}")
            self.risk_engine = None
            self.differential = None
            self.guidelines = None
            self.calibrator = None
            self.report_generator = None
        
        # ========== Clinical Override Rules ==========
        self.override_rules = getattr(Config, 'CLINICAL_OVERRIDE_RULES', {})


    # ========================================================================
    # ADD THESE METHODS AFTER _init_components
    # ========================================================================
    
    def get_model_status(self):
        """Get status of all models"""
        status = {
            'segmenter': 'Active' if self.segmenter else 'Inactive',
            'semantic': 'Active' if self.semantic else 'Inactive',
            'risk_engine': 'Active' if self.risk_engine else 'Inactive',
            'differential': 'Active' if self.differential else 'Inactive',
            'guidelines': 'Active' if self.guidelines else 'Inactive',
            'calibrator': 'Active' if self.calibrator else 'Inactive',
            'quality_assessor': 'Active' if self.quality_assessor else 'Inactive',
            'safety_checker': 'Active' if self.safety_checker else 'Inactive'
        }
        return status
    
    # ========================================================================
    # AGGREGATE FEATURES
    # ========================================================================
    def _aggregate_features(self, all_features: List[Dict]) -> Dict:
        """Aggregate features from multiple lesions"""
        if not all_features:
            return {
                'area': 0,
                'circularity': 1.0,
                'solidity': 1.0,
                'perimeter': 0,
                'aspect_ratio': 1.0,
                'extent': 0,
                'white_ratio': 0,
                'red_ratio': 0,
                'hue_var': 0,
                'sat_var': 0,
                'lesion_count': 0,
                'irregular_count': 0
            }
        
        areas = [f.get('area', 0) for f in all_features]
        total_area = sum(areas) or 1
        
        aggregated = {
            'area': float(np.mean(areas)),
            'circularity': float(np.average([f.get('circularity', 1) for f in all_features], weights=areas)),
            'solidity': float(np.average([f.get('solidity', 1) for f in all_features], weights=areas)),
            'perimeter': float(np.mean([f.get('perimeter', 0) for f in all_features])),
            'aspect_ratio': float(np.mean([f.get('aspect_ratio', 1) for f in all_features])),
            'extent': float(np.mean([f.get('extent', 0) for f in all_features])),
            'white_ratio': float(np.average([f.get('white_ratio', 0) for f in all_features], weights=areas)),
            'red_ratio': float(np.average([f.get('red_ratio', 0) for f in all_features], weights=areas)),
            'hue_var': float(np.mean([f.get('hue_var', 0) for f in all_features])),
            'sat_var': float(np.mean([f.get('sat_var', 0) for f in all_features])),
            'lesion_count': len(all_features),
            'irregular_count': sum(1 for f in all_features if f.get('circularity', 1) < 0.5)
        }
        
        return aggregated
    
    def analyze(self, image, patient_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete clinical analysis pipeline
        """
        start_time = datetime.now()
        
        result = {
            'success': False,
            'timestamp': start_time.isoformat(),
            'patient_id': patient_id or 'N/A',
            'processing_time': 0,
            'lesions': [],
            'lesion_count': 0,
            'risk_score': 0.0,
            'risk_level': 'low',
            'risk_message': '',
            'clinical_terms': [],
            'recommendation': '',
            'explanation': '',
            'uncertainty': 0.0,
            'differential': [],
            'guidelines': [],
            'multi_labels': {},
            'tracking': None,
            'report': '',
            'clinical_features': {},
            'quality': {},
            'safety': {}
        }
        
        try:
            # ================================================================
            # STEP 1: IMAGE QUALITY ASSESSMENT
            # ================================================================
            if self.quality_assessor:
                quality = self.quality_assessor.assess(image)
                result['quality'] = quality
                if not quality.get('passed', False):
                    result['error'] = f"Image quality issues: {', '.join(quality.get('issues', []))}"
                    return result
            else:
                result['quality'] = {'passed': True, 'issues': []}
            
            # ================================================================
            # STEP 2: SAFETY CHECK (OOD Detection)
            # ================================================================
            if self.safety_checker:
                safety = self.safety_checker.check(image)
                result['safety'] = safety
                if not safety.get('passed', False):
                    result['error'] = f"Safety check failed: {safety.get('reason', 'Unknown')}"
                    return result
            else:
                result['safety'] = {'passed': True, 'reason': 'Safety check skipped'}
            
            # ================================================================
            # STEP 3: SEGMENTATION
            # ================================================================
            if self.segmenter:
                lesions = self.segmenter.detect_lesions(image)
                result['lesions'] = lesions
                result['lesion_count'] = len(lesions)
            else:
                lesions = []
                result['lesions'] = []
                result['lesion_count'] = 0
            
            # ================================================================
            # STEP 4: SEMANTIC ANALYSIS
            # ================================================================
            if self.semantic:
                semantic = self.semantic.predict_ensemble(image, return_heatmap=False)
                result['semantic'] = semantic
            else:
                semantic = None
                result['semantic'] = None
            
            # ================================================================
            # STEP 5: FEATURE EXTRACTION
            # ================================================================
            all_features = []
            if self.mask_feature_extractor:
                for lesion in lesions[:5]:
                    if lesion.get('segmentation') is not None:
                        try:
                            geo_features = self.mask_feature_extractor.extract(
                                lesion['segmentation'],
                                np.array(image)
                            )
                            lesion['features'] = geo_features
                            all_features.append(geo_features)
                        except Exception as e:
                            print(f"⚠️ Feature extraction error: {e}")
                            continue
            
            # Aggregate features
            aggregated_features = self._aggregate_features(all_features)
            result['features'] = aggregated_features
            
            # Extract clinical features
            if self.clinical_feature_extractor:
                clinical_features = self.clinical_feature_extractor.extract(
                    aggregated_features,
                    semantic,
                    lesions
                )
                result['clinical_features'] = clinical_features
            else:
                clinical_features = {'terms': []}
                result['clinical_features'] = clinical_features
            
            # ================================================================
            # STEP 6: CLINICAL OVERRIDE
            # ================================================================
            override_result = self._apply_clinical_override(
                lesions, 
                semantic, 
                aggregated_features,
                clinical_features
            )
            result['clinical_override'] = override_result
            
            # ================================================================
            # STEP 7: RISK CALCULATION
            # ================================================================
            if self.risk_engine:
                risk_score = self.risk_engine.calculate(
                    aggregated_features=aggregated_features,
                    clinical_features=clinical_features,
                    semantic=semantic,
                    override=override_result,
                    calibrated={}
                )
                result['risk_score'] = float(risk_score)
                result['risk_level'] = self.risk_engine.get_level(risk_score)
                result['risk_message'] = self.risk_engine.get_message(risk_score)
            else:
                risk_score = 0.3
                result['risk_score'] = 0.3
                result['risk_level'] = 'low'
                result['risk_message'] = 'Risk calculation unavailable'
            
            # ================================================================
            # STEP 8: DIFFERENTIAL DIAGNOSIS
            # ================================================================
            if self.differential:
                differential = self.differential.diagnose(
                    clinical_features,
                    risk_score,
                    aggregated_features
                )
                result['differential'] = differential
            else:
                result['differential'] = []
            
            # ================================================================
            # STEP 9: GUIDELINE EVALUATION
            # ================================================================
            if self.guidelines:
                guidelines = self.guidelines.evaluate(clinical_features)
                result['guidelines'] = guidelines
            else:
                result['guidelines'] = []
            
            # ================================================================
            # STEP 10: MULTI-LABEL CLASSIFICATION
            # ================================================================
            multi_labels = self._multi_label_classify(clinical_features)
            result['multi_labels'] = multi_labels
            
            # ================================================================
            # STEP 11: CLINICAL RECOMMENDATION
            # ================================================================
            recommendation = self._generate_recommendation(
                risk_score=risk_score,
                differential=result.get('differential', []),
                guidelines=result.get('guidelines', []),
                override=override_result,
                clinical_features=clinical_features
            )
            result['recommendation'] = recommendation
            
            # ================================================================
            # STEP 12: EXPLANATION
            # ================================================================
            explanation = self._generate_explanation(
                risk_score=risk_score,
                clinical_features=clinical_features,
                override=override_result,
                differential=result.get('differential', [])
            )
            result['explanation'] = explanation
            
            # ================================================================
            # STEP 13: UNCERTAINTY
            # ================================================================
            uncertainty = self._calculate_uncertainty(
                semantic=semantic,
                lesions=lesions,
                clinical_features=clinical_features
            )
            result['uncertainty'] = float(uncertainty)
            
            # ================================================================
            # STEP 14: LESION TRACKING
            # ================================================================
            if patient_id and aggregated_features:
                tracking = self._track_lesion(patient_id, aggregated_features, clinical_features)
                result['tracking'] = tracking
            else:
                result['tracking'] = None
            
            # ================================================================
            # STEP 15: REPORT GENERATION
            # ================================================================
            if self.report_generator:
                report_data = self._prepare_report_data(result)
                try:
                    result['report'] = self.report_generator.generate_html(report_data)
                except Exception as e:
                    print(f"⚠️ Report generation failed: {e}")
                    result['report'] = f"Report generation failed: {e}"
            else:
                result['report'] = ''
            
            # ================================================================
            # STEP 16: CLINICAL TERMS
            # ================================================================
            result['clinical_terms'] = clinical_features.get('terms', [])
            
            # ================================================================
            # FINAL
            # ================================================================
            result['success'] = True
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result['success'] = False
            result['error'] = str(e)
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            return result
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _aggregate_features(self, all_features: List[Dict]) -> Dict:
        """Aggregate features from multiple lesions"""
        if not all_features:
            return {
                'area': 0,
                'circularity': 1.0,
                'solidity': 1.0,
                'perimeter': 0,
                'aspect_ratio': 1.0,
                'extent': 0,
                'white_ratio': 0,
                'red_ratio': 0,
                'hue_var': 0,
                'sat_var': 0,
                'lesion_count': 0,
                'irregular_count': 0
            }
        
        areas = [f.get('area', 0) for f in all_features]
        total_area = sum(areas) or 1
        
        aggregated = {
            'area': float(np.mean(areas)),
            'circularity': float(np.average([f.get('circularity', 1) for f in all_features], weights=areas)),
            'solidity': float(np.average([f.get('solidity', 1) for f in all_features], weights=areas)),
            'perimeter': float(np.mean([f.get('perimeter', 0) for f in all_features])),
            'aspect_ratio': float(np.mean([f.get('aspect_ratio', 1) for f in all_features])),
            'extent': float(np.mean([f.get('extent', 0) for f in all_features])),
            'white_ratio': float(np.average([f.get('white_ratio', 0) for f in all_features], weights=areas)),
            'red_ratio': float(np.average([f.get('red_ratio', 0) for f in all_features], weights=areas)),
            'hue_var': float(np.mean([f.get('hue_var', 0) for f in all_features])),
            'sat_var': float(np.mean([f.get('sat_var', 0) for f in all_features])),
            'lesion_count': len(all_features),
            'irregular_count': sum(1 for f in all_features if f.get('circularity', 1) < 0.5)
        }
        
        return aggregated
    
    def _apply_clinical_override(self, lesions, semantic, features, clinical_features):
        """Apply clinical override rules"""
        override = {
            'applied': False,
            'reason': None,
            'override_risk': None,
            'rules_triggered': []
        }
        
        if not lesions and not semantic:
            return override
        
        # Rule 1: Lesion detected but semantic says healthy
        if len(lesions) > 0:
            semantic_label = semantic.get('ensemble', {}).get('label', '') if semantic else ''
            if semantic_label == 'healthy mouth':
                override['applied'] = True
                override['reason'] = 'Lesion detected but semantic model indicates healthy'
                override['override_risk'] = 0.5
                override['rules_triggered'].append('lesion_with_healthy')
        
        # Rule 2: Multiple irregular lesions
        irregular_count = features.get('irregular_count', 0)
        if irregular_count >= 2:
            override['applied'] = True
            override['reason'] = f'Multiple irregular lesions detected ({irregular_count})'
            override['override_risk'] = 0.7
            override['rules_triggered'].append('multiple_irregular')
        
        # Rule 3: Large white plaque
        white_ratio = features.get('white_ratio', 0)
        area = features.get('area', 0)
        if white_ratio > 0.4 and area > 100:
            override['applied'] = True
            override['reason'] = 'Large white plaque with suspicious features'
            override['override_risk'] = 0.65
            override['rules_triggered'].append('large_white_plaque')
        
        return override
    
    def _multi_label_classify(self, clinical_features: Dict) -> Dict:
        """Multi-label classification"""
        return {
            'ulcer': bool(clinical_features.get('ulcer', False)),
            'white_plaque': bool(clinical_features.get('white_plaque', False)),
            'irregular_border': bool(clinical_features.get('irregular_border', False)),
            'bleeding': bool(clinical_features.get('bleeding', False)),
            'erythema': bool(clinical_features.get('red_ratio', 0) > 0.3),
            'swelling': bool(clinical_features.get('area', 0) > 100),
            'large_lesion': bool(clinical_features.get('area', 0) > 80),
            'high_risk_location': bool(clinical_features.get('location', '') in getattr(Config, 'HIGH_RISK_LOCATIONS', []))
        }
    
    def _generate_recommendation(self, risk_score, differential, guidelines, override, clinical_features):
        """Generate clinical recommendation"""
        # Priority 1: Cancer suspicion
        if differential and len(differential) > 0:
            top = differential[0]
            if top.get('condition') == 'Squamous Cell Carcinoma' and top.get('score', 0) > 0.3:
                return "🚨 IMMEDIATE BIOPSY AND ONCOLOGY REFERRAL"
        
        # Priority 2: Override triggered
        if override.get('applied'):
            return f"⚠️ Clinical override: {override.get('reason', 'Unknown')}. Specialist evaluation recommended"
        
        # Priority 3: High risk
        if risk_score > 0.7:
            return "🔴 URGENT SPECIALIST REFERRAL within 1 week"
        
        # Priority 4: Medium risk
        if risk_score > 0.4:
            if differential and len(differential) > 1:
                top = differential[0]
                return f"🟡 Consider {top.get('condition', 'Unknown')}. Schedule specialist evaluation within 2 weeks"
            return "🟡 Schedule dental evaluation within 1 month"
        
        # Priority 5: Low risk with lesions
        if clinical_features.get('lesion_count', 0) > 0:
            return "🟢 Monitor lesion. Follow-up in 3-6 months"
        
        # Priority 6: Normal
        return "🟢 Routine dental check-up in 6 months"
    
    def _generate_explanation(self, risk_score, clinical_features, override, differential):
        """Generate feature-based explanation"""
        parts = []
        
        # Lesion findings
        lesion_count = clinical_features.get('lesion_count', 0)
        if lesion_count > 0:
            parts.append(f"{lesion_count} lesion(s) detected")
            if clinical_features.get('irregular_border'):
                parts.append("Irregular border detected")
            if clinical_features.get('white_plaque'):
                parts.append("White plaque present")
            area = clinical_features.get('area', 0)
            if area > 80:
                parts.append(f"Large lesion ({area:.0f} px²)")
        else:
            parts.append("No lesions detected")
        
        # Override
        if override.get('applied'):
            parts.append(f"⚠️ {override.get('reason', 'Unknown')}")
        
        # Differential
        if differential and len(differential) > 0:
            top = differential[0]
            parts.append(f"Primary differential: {top.get('condition', 'Unknown')}")
        
        # Risk level
        if risk_score > 0.7:
            parts.append("🔴 HIGH RISK - Immediate attention")
        elif risk_score > 0.4:
            parts.append("🟡 MODERATE RISK - Follow-up needed")
        else:
            parts.append("🟢 LOW RISK - Routine care")
        
        return " | ".join(parts)
    
    def _calculate_uncertainty(self, semantic, lesions, clinical_features):
        """Calculate overall uncertainty"""
        uncertainty_scores = []
        
        # Semantic uncertainty
        if semantic and 'ensemble' in semantic:
            uncertainty_scores.append(semantic['ensemble'].get('uncertainty', 0.5))
        
        # Lesion confidence
        if lesions:
            confidences = [l.get('confidence', 0.5) for l in lesions]
            avg_conf = float(np.mean(confidences))
            uncertainty_scores.append(1 - avg_conf)
        
        # Feature ambiguity
        white_ratio = clinical_features.get('white_ratio', 0)
        if 0.15 < white_ratio < 0.25:
            uncertainty_scores.append(0.3)
        
        if uncertainty_scores:
            return float(np.mean(uncertainty_scores))
        return 0.5
    
    def _track_lesion(self, patient_id, features, clinical_features):
        """Track lesion changes over time"""
        if patient_id not in self.tracking_history:
            self.tracking_history[patient_id] = []
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'area': features.get('area', 0),
            'lesion_count': features.get('lesion_count', 0),
            'white_ratio': features.get('white_ratio', 0),
            'circularity': features.get('circularity', 1)
        }
        self.tracking_history[patient_id].append(entry)
        
        # Keep only recent
        max_history = getattr(Config, 'MAX_TRACKING_HISTORY', 10)
        if len(self.tracking_history[patient_id]) > max_history:
            self.tracking_history[patient_id] = self.tracking_history[patient_id][-max_history:]
        
        # Compare with previous
        if len(self.tracking_history[patient_id]) >= 2:
            prev = self.tracking_history[patient_id][-2]
            curr = self.tracking_history[patient_id][-1]
            
            prev_area = prev.get('area', 0)
            curr_area = curr.get('area', 0)
            
            if prev_area > 0:
                area_change_pct = ((curr_area / prev_area) - 1) * 100
            else:
                area_change_pct = 0
            
            threshold = getattr(Config, 'TRACKING_CHANGE_THRESHOLD', 0.2)
            
            return {
                'size_change': curr_area - prev_area,
                'size_percentage': area_change_pct,
                'has_grown': curr_area > prev_area,
                'significantly_changed': abs(area_change_pct / 100) > threshold,
                'previous_date': prev.get('timestamp', 'N/A'),
                'current_date': curr.get('timestamp', 'N/A'),
                'previous_area': prev_area,
                'current_area': curr_area,
                'previous_lesion_count': prev.get('lesion_count', 0),
                'current_lesion_count': curr.get('lesion_count', 0)
            }
        
        return None
        
    def _prepare_report_data(self, result: Dict) -> Dict:
        """Prepare data for report generation"""
        tracking = result.get('tracking')
        if tracking and isinstance(tracking, dict):
            # Convert tracking values to safe types
            for key in ['size_percentage', 'size_change', 'previous_area', 'current_area']:
                if key in tracking and tracking[key] is not None:
                    try:
                        tracking[key] = float(tracking[key])
                    except (ValueError, TypeError):
                        tracking[key] = 0.0
        
        return {
            'patient_id': result.get('patient_id', 'N/A'),
            'date': result.get('timestamp', datetime.now().isoformat()),
            'lesion_count': result.get('lesion_count', 0),
            'area': result.get('features', {}).get('area', 0),
            'circularity': result.get('features', {}).get('circularity', 0),
            'white_ratio': result.get('features', {}).get('white_ratio', 0),
            'risk_score': result.get('risk_score', 0),
            'risk_level': result.get('risk_level', 'low'),
            'risk_message': result.get('risk_message', ''),
            'clinical_terms': ', '.join(result.get('clinical_terms', [])),
            'differential': result.get('differential', []),
            'recommendation': result.get('recommendation', 'N/A'),
            'explanation': result.get('explanation', ''),
            'uncertainty': result.get('uncertainty', 0),
            'tracking': tracking
        }
    
    def get_model_status(self):
        """Get status of all models"""
        status = {
            'segmenter': 'Active' if self.segmenter else 'Inactive',
            'semantic': 'Active' if self.semantic else 'Inactive',
            'risk_engine': 'Active' if self.risk_engine else 'Inactive',
            'differential': 'Active' if self.differential else 'Inactive',
            'guidelines': 'Active' if self.guidelines else 'Inactive',
            'calibrator': 'Active' if self.calibrator else 'Inactive',
            'quality_assessor': 'Active' if self.quality_assessor else 'Inactive',
            'safety_checker': 'Active' if self.safety_checker else 'Inactive'
        }
        return status

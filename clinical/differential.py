"""
Probabilistic Differential Diagnosis Engine
"""

from typing import List, Dict, Any


class DifferentialDiagnosis:
    """Generate probabilistic differential diagnoses"""
    
    def __init__(self):
        self.conditions = {
            'Squamous Cell Carcinoma': {
                'features': ['ulcer', 'irregular_border', 'bleeding', 'swelling', 'large_lesion'],
                'base_risk': 0.8,
                'description': 'Malignant lesion with warning features',
                'recommendation': 'Immediate biopsy',
                'weight': 1.0
            },
            'Leukoplakia': {
                'features': ['white_plaque', 'irregular_border'],
                'base_risk': 0.3,
                'description': 'White plaque that cannot be scraped off',
                'recommendation': 'Biopsy if no improvement',
                'weight': 0.8
            },
            'Oral Lichen Planus': {
                'features': ['white_plaque', 'erythema', 'ulcer'],
                'base_risk': 0.2,
                'description': 'Reticular white lesions with inflammation',
                'recommendation': 'Dermatology consultation',
                'weight': 0.6
            },
            'Traumatic Ulcer': {
                'features': ['ulcer', 'bleeding'],
                'base_risk': 0.1,
                'description': 'Ulcer from trauma or friction',
                'recommendation': 'Identify and remove traumatic cause',
                'weight': 0.5
            },
            'Gingivitis': {
                'features': ['redness', 'swelling'],
                'base_risk': 0.15,
                'description': 'Gum inflammation',
                'recommendation': 'Professional cleaning',
                'weight': 0.4
            },
            'Periodontitis': {
                'features': ['swelling', 'bleeding', 'large_lesion'],
                'base_risk': 0.4,
                'description': 'Gum disease with bone loss',
                'recommendation': 'Periodontal therapy',
                'weight': 0.7
            }
        }
    
    def diagnose(self, clinical_features: Dict, risk_score: float, aggregated_features: Dict) -> List[Dict]:
        """Generate differential diagnosis"""
        results = []
        
        for condition, info in self.conditions.items():
            feature_score = self._calculate_match(clinical_features, info['features'])
            risk_factor = (risk_score + info['base_risk']) / 2
            uncertainty = clinical_features.get('uncertainty', 0.3)
            uncertainty_factor = 1 - uncertainty * 0.3
            
            final_score = (
                feature_score * 0.4 +
                risk_factor * 0.4 +
                uncertainty_factor * 0.1 +
                info['weight'] * 0.1
            )
            final_score = min(1.0, max(0.0, final_score))
            
            results.append({
                'condition': condition,
                'score': final_score,
                'risk': info['base_risk'],
                'description': info['description'],
                'recommendation': info['recommendation'],
                'feature_match': feature_score
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Normalize
        total = sum(r['score'] for r in results) or 1
        for r in results:
            r['score'] = r['score'] / total
        
        return results
    
    def _calculate_match(self, features: Dict, required: List[str]) -> float:
        """Calculate match score"""
        if not required:
            return 0
        
        feature_map = {
            'ulcer': features.get('ulcer', False),
            'white_plaque': features.get('white_plaque', False),
            'irregular_border': features.get('irregular_border', False),
            'bleeding': features.get('bleeding', False),
            'swelling': features.get('swelling', False),
            'erythema': features.get('red_ratio', 0) > 0.3,
            'large_lesion': features.get('area', 0) > 80,
            'redness': features.get('red_ratio', 0) > 0.3
        }
        
        matched = sum(1 for f in required if feature_map.get(f, False))
        return matched / len(required) if required else 0

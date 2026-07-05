"""
Clinical Guidelines Engine
"""

from typing import List, Dict, Any
from config import Config


class GuidelineEngine:
    """Evaluate clinical guidelines"""
    
    def __init__(self):
        self.rules = [
            {
                'condition': lambda x: x.get('ulcer', False) and x.get('duration', 0) > 14,
                'action': 'urgent_referral',
                'message': 'Ulcer persisting >14 days - urgent specialist referral',
                'severity': 'high',
                'source': 'WHO Guidelines'
            },
            {
                'condition': lambda x: x.get('white_plaque', False) and x.get('area', 0) > 100,
                'action': 'biopsy',
                'message': 'Large white plaque (>100 px²) - biopsy recommended',
                'severity': 'high',
                'source': 'Oral Cancer Screening'
            },
            {
                'condition': lambda x: x.get('irregular_border', False) and x.get('bleeding', False),
                'action': 'high_risk',
                'message': 'Irregular border with bleeding - high risk feature',
                'severity': 'high',
                'source': 'Malignancy Indicators'
            },
            {
                'condition': lambda x: x.get('location', '') in getattr(Config, 'HIGH_RISK_LOCATIONS', []),
                'action': 'high_risk_location',
                'message': 'Lesion in high-risk location - requires careful evaluation',
                'severity': 'high',
                'source': 'High-Risk Oral Sites'
            },
            {
                'condition': lambda x: x.get('area', 0) > 50 and x.get('duration', 0) > 7,
                'action': 'monitor',
                'message': 'Medium lesion (>50 px²) persisting >7 days - monitor closely',
                'severity': 'medium',
                'source': 'Clinical Follow-up'
            }
        ]
    
    def evaluate(self, features: Dict) -> List[Dict]:
        """Evaluate guidelines"""
        results = []
        for rule in self.rules:
            try:
                if rule['condition'](features):
                    results.append({
                        'action': rule['action'],
                        'message': rule['message'],
                        'severity': rule['severity'],
                        'source': rule.get('source', 'Clinical Guidelines')
                    })
            except Exception:
                continue
        
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        results.sort(key=lambda x: severity_order.get(x['severity'], 3))
        return results

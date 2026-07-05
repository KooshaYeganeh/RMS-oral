"""
Report Generation
"""

from datetime import datetime
from typing import Dict, Any, List
import json


class ReportGenerator:
    """Generate clinical reports"""
    
    def _safe_str(self, value):
        if value is None:
            return 'N/A'
        if isinstance(value, (bool, int, float)):
            return str(value)
        if isinstance(value, (list, dict)):
            try:
                return json.dumps(value)
            except:
                return str(value)
        return str(value)
    
    def generate_html(self, data: Dict) -> str:
        """Generate HTML report"""
        patient_id = self._safe_str(data.get('patient_id', 'N/A'))
        date = self._safe_str(data.get('date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        risk_score = float(data.get('risk_score', 0))
        risk_level = self._safe_str(data.get('risk_level', 'low'))
        risk_message = self._safe_str(data.get('risk_message', ''))
        lesion_count = int(data.get('lesion_count', 0))
        area = float(data.get('area', 0))
        circularity = float(data.get('circularity', 0))
        white_ratio = float(data.get('white_ratio', 0))
        clinical_terms = self._safe_str(data.get('clinical_terms', 'N/A'))
        explanation = self._safe_str(data.get('explanation', 'N/A'))
        recommendation = self._safe_str(data.get('recommendation', 'N/A'))
        uncertainty = float(data.get('uncertainty', 0))
        
        risk_colors = {'high': '#dc2626', 'medium': '#f59e0b', 'low': '#22c55e'}
        risk_color = risk_colors.get(risk_level, '#94a3b8')
        
        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: auto; padding: 20px;">
            <h2 style="color: #0ea5e9;">📋 Clinical Report</h2>
            <p><strong>Patient:</strong> {patient_id}</p>
            <p><strong>Date:</strong> {date}</p>
            <hr>
            
            <h3>📊 Risk Assessment</h3>
            <p><strong>Risk Score:</strong> {risk_score:.2f}</p>
            <p><strong>Risk Level:</strong> <span style="color: {risk_color}">{risk_level.upper()}</span></p>
            <p>{risk_message}</p>
            
            <h3>🔍 Lesion Findings</h3>
            <p><strong>Lesions Detected:</strong> {lesion_count}</p>
            <p><strong>Area:</strong> {area:.0f} px²</p>
            <p><strong>Circularity:</strong> {circularity:.2f}</p>
            <p><strong>White Ratio:</strong> {white_ratio:.2f}</p>
            <p><strong>Uncertainty:</strong> {uncertainty:.2f}</p>
            
            <h3>🧠 AI Analysis</h3>
            <p><strong>Clinical Terms:</strong> {clinical_terms}</p>
            <p><strong>Explanation:</strong> {explanation}</p>
            
            <h3>🩺 Differential Diagnosis</h3>
            {self._format_differential(data.get('differential', []))}
            
            <h3>💡 Recommendation</h3>
            <p>{recommendation}</p>
            
            <hr>
            <p style="color: #94a3b8; font-size: 0.85rem;">⚠️ This report is for clinical decision support only</p>
        </div>
        """
        return html
    
    def _format_differential(self, differential: List[Dict]) -> str:
        """Format differential diagnosis"""
        if not differential:
            return "<p>No differential diagnosis recorded</p>"
        
        html = "<ul>"
        for item in differential[:5]:
            score = float(item.get('score', 0))
            condition = self._safe_str(item.get('condition', 'Unknown'))
            recommendation = self._safe_str(item.get('recommendation', ''))
            html += f"<li><strong>{condition}</strong> (Match: {score:.0%}) - {recommendation}</li>"
        html += "</ul>"
        return html

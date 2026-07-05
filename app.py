"""
RMS-ORAL: Clinical Decision System - Production App
"""

import os
import sys
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from clinical.decision_engine import ClinicalDecisionEngine
from clinical.safety import SafetyChecker
from utils.image_quality import ImageQualityAssessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
quality_assessor = ImageQualityAssessor()
safety_checker = SafetyChecker()
decision_engine = ClinicalDecisionEngine()


# =========================
# 🔄 JSON SERIALIZER HELPERS
# =========================
def convert_to_serializable(obj):
    """Convert non-serializable objects to JSON serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return str(obj)


def safe_jsonify(data):
    """Safely convert any data to JSON"""
    try:
        return jsonify(convert_to_serializable(data))
    except Exception as e:
        return jsonify({"error": f"Serialization error: {str(e)}", "data": str(data)})


@app.route("/", methods=["GET", "POST"])
def index():
    """Main interface"""
    result = None
    error = None
    
    try:
        if request.method == "POST":
            file = request.files.get("image")
            if not file:
                return jsonify({"error": "no image"}), 400
            
            # Save and load
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            image = Image.open(path).convert("RGB")
            
            # Get patient ID
            patient_id = request.form.get("patient_id")
            
            # Analyze
            analysis = decision_engine.analyze(image, patient_id)
            
            if analysis.get('success', False):
                # Convert all data to JSON serializable
                safe_analysis = convert_to_serializable(analysis)
                
                # Handle tracking safely
                tracking = safe_analysis.get('tracking')
                if tracking and isinstance(tracking, dict):
                    # Ensure all values are numbers
                    for key in ['size_percentage', 'size_change', 'previous_area', 'current_area']:
                        if key in tracking and tracking[key] is not None:
                            try:
                                tracking[key] = float(tracking[key])
                            except (ValueError, TypeError):
                                tracking[key] = 0.0
                
                result = {
                    'risk_score': safe_analysis.get('risk_score', 0),
                    'risk_level': safe_analysis.get('risk_level', 'low'),
                    'risk_message': safe_analysis.get('risk_message', ''),
                    'lesion_count': safe_analysis.get('lesion_count', 0),
                    'clinical_terms': ', '.join(safe_analysis.get('clinical_features', {}).get('terms', [])),
                    'recommendation': safe_analysis.get('recommendation', ''),
                    'explanation': safe_analysis.get('explanation', ''),
                    'uncertainty': safe_analysis.get('uncertainty', 0),
                    'differential': safe_analysis.get('differential', []),
                    'guidelines': safe_analysis.get('guidelines', []),
                    'multi_labels': safe_analysis.get('multi_labels', {}),
                    'tracking': tracking,
                    'report': safe_analysis.get('report', ''),
                    'detailed': safe_analysis
                }
            else:
                error = analysis.get('error', 'Analysis failed')
                
    except Exception as e:
        error = str(e)
        import traceback
        traceback.print_exc()
    
    return render_template("index.html", result=result, error=error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint"""
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "no image"}), 400
        
        image = Image.open(file).convert("RGB")
        patient_id = request.form.get("patient_id")
        
        result = decision_engine.analyze(image, patient_id)
        return safe_jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check"""
    return {
        "status": "ok",
        "device": str(Config.DEVICE),
        "models": decision_engine.get_model_status(),
        "timestamp": datetime.now().isoformat()
    }


@app.route("/api/track/<patient_id>", methods=["GET"])
def api_track(patient_id):
    """Get patient tracking history"""
    history = decision_engine.tracking_history.get(patient_id, [])
    return safe_jsonify({
        "success": True,
        "patient_id": patient_id,
        "history": convert_to_serializable(history),
        "count": len(history)
    })


if __name__ == "__main__":
    print("=" * 60)
    print("🦷 RMS-ORAL: Clinical Decision System")
    print("=" * 60)
    print(f"📱 Running on: http://0.0.0.0:5000")
    print(f"🔧 Device: {Config.DEVICE}")
    print(f"📊 Status: Production Ready")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5005, debug=False)

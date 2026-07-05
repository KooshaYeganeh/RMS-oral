"""
Semantic Ensemble - BioMedCLIP + Clinical ViT
"""

import os
import torch
import numpy as np
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SemanticEnsemble:
    """
    Ensemble of semantic models
    """
    
    def __init__(self, vit_model_path=None):
        self.models = {}
        self.processors = {}
        self.tokenizers = {}
        self.labels = [
            "healthy mouth", "dental caries", "gingivitis",
            "periodontitis", "oral lesion", "oral cancer suspicion"
        ]
        self.num_classes = len(self.labels)
        self.vit_model_path = vit_model_path
        self._loaded = False
    
    def load_biomedclip(self):
        """Load BioMedCLIP"""
        try:
            import open_clip
            
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            
            self.models['biomedclip'] = {
                'model': model, 
                'type': 'clip',
                'num_classes': self.num_classes
            }
            self.processors['biomedclip'] = preprocess
            self.tokenizers['biomedclip'] = tokenizer
            
            print("✅ BioMedCLIP loaded")
            return True
        except Exception as e:
            print(f"⚠️ BioMedCLIP failed: {e}")
            return False
    
    def load_clinical_vit(self):
        """Load Clinical ViT"""
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
            
            model_path = self.vit_model_path
            
            if model_path and os.path.exists(model_path):
                try:
                    processor = ViTImageProcessor.from_pretrained(model_path)
                    model = ViTForImageClassification.from_pretrained(
                        model_path, 
                        use_safetensors=True, 
                        ignore_mismatched_sizes=True
                    )
                    vit_num_classes = model.classifier.out_features if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features') else 2
                    self.models['clinical_vit'] = {
                        'model': model, 
                        'type': 'vit',
                        'num_classes': vit_num_classes
                    }
                    self.processors['clinical_vit'] = processor
                    print(f"✅ Fine-tuned ViT loaded ({vit_num_classes} classes)")
                    return True
                except Exception as e:
                    print(f"⚠️ Fine-tuned ViT failed: {e}")
            
            # Fallback to generic
            print("⚠️ Using generic ViT")
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
            self.models['clinical_vit'] = {
                'model': model, 
                'type': 'vit',
                'num_classes': 1000
            }
            self.processors['clinical_vit'] = processor
            print("✅ Generic ViT loaded")
            return True
            
        except Exception as e:
            print(f"⚠️ Clinical ViT failed: {e}")
            return False
    
    def predict_ensemble(self, image, return_heatmap=False):
        """Get ensemble prediction"""
        results = {}
        
        for name, info in self.models.items():
            result = self._predict_single(image, name)
            if result:
                results[name] = result
        
        if not results:
            return None
        
        # Weighted ensemble
        total_weight = sum(r.get('confidence', 0.5) for r in results.values())
        if total_weight > 0:
            ensemble_probs = np.zeros(self.num_classes)
            for r in results.values():
                weight = r.get('confidence', 0.5) / total_weight
                probs = np.array(r.get('probs', np.ones(self.num_classes) / self.num_classes))
                if len(probs) != self.num_classes:
                    temp = np.zeros(self.num_classes)
                    min_len = min(len(probs), self.num_classes)
                    temp[:min_len] = probs[:min_len]
                    temp = temp / temp.sum() if temp.sum() > 0 else np.ones(self.num_classes) / self.num_classes
                    probs = temp
                ensemble_probs += probs * weight
            
            ensemble_probs = ensemble_probs / ensemble_probs.sum()
            top1_idx = ensemble_probs.argmax()
            
            return {
                'ensemble': {
                    'label': self.labels[top1_idx],
                    'confidence': float(ensemble_probs[top1_idx]),
                    'uncertainty': self._calculate_uncertainty(ensemble_probs),
                    'probs': ensemble_probs.tolist()
                },
                'individual': results
            }
        
        return None
    
    def _predict_single(self, image, model_name):
        """Predict with single model"""
        try:
            info = self.models[model_name]
            model = info['model']
            model_type = info['type']
            
            if model_type == 'clip':
                device = "cuda" if torch.cuda.is_available() else "cpu"
                image_input = self.processors[model_name](image).unsqueeze(0).to(device)
                text_inputs = self.tokenizers[model_name](self.labels).to(device)
                
                with torch.no_grad():
                    img_f = model.encode_image(image_input)
                    txt_f = model.encode_text(text_inputs)
                    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                    logits = img_f @ txt_f.T
                    probs = torch.softmax(logits, dim=-1)[0]
                    
                    top1_idx = probs.argmax().item()
                    return {
                        'label': self.labels[top1_idx],
                        'confidence': probs[top1_idx].item(),
                        'probs': probs.cpu().numpy().tolist(),
                        'type': 'clip'
                    }
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = self.processors[model_name](image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)[0]
                    
                    # Map to our labels
                    num_classes = info.get('num_classes', 1000)
                    mapped_probs = np.zeros(self.num_classes)
                    
                    for i in range(min(num_classes, self.num_classes)):
                        mapped_probs[i] = probs[i].item()
                    
                    if mapped_probs.sum() > 0:
                        mapped_probs = mapped_probs / mapped_probs.sum()
                    else:
                        mapped_probs = np.ones(self.num_classes) / self.num_classes
                    
                    top1_idx = mapped_probs.argmax()
                    return {
                        'label': self.labels[top1_idx],
                        'confidence': float(mapped_probs[top1_idx]),
                        'probs': mapped_probs.tolist(),
                        'type': 'vit'
                    }
        except Exception as e:
            print(f"⚠️ {model_name} prediction failed: {e}")
            return None
    
    def _calculate_uncertainty(self, probs):
        """Calculate uncertainty using entropy"""
        probs = np.clip(probs, 1e-8, 1.0)
        entropy = -(probs * np.log(probs)).sum()
        max_entropy = np.log(len(probs))
        return float(min(1.0, max(0.0, entropy / max_entropy if max_entropy > 0 else 0)))
    
    def get_model_status(self):
        """Get status of models"""
        return {name: info['type'] for name, info in self.models.items()}
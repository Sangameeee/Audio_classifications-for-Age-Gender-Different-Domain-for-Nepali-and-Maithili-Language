import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json

class DomainClassifier:
    def __init__(self, model_dir, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        with open(os.path.join(model_dir, "label_mappings.json"), "r", encoding="utf-8") as f:
            loaded_label_mappings = json.load(f)
        self.id2label = {int(k): v for k, v in loaded_label_mappings["id2label"].items()}
        self.label2id = loaded_label_mappings["label2id"]
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, threshold=0.5, max_length=256):
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        domain_probs = {self.id2label[i]: float(probabilities[i]) for i in range(len(self.id2label))}
        pred_id = np.argmax(probabilities)
        pred_label = self.id2label[pred_id]
        pred_prob = float(probabilities[pred_id])
        if pred_prob < threshold:
            final_domain = "General"
            thresholded = True
        else:
            final_domain = pred_label
            thresholded = False
        return {
            "text": text[:80] + "..." if len(text) > 80 else text,
            "predicted_domain": pred_label,
            "final_domain": final_domain,
            "confidence": pred_prob,
            "thresholded": thresholded,
            "probabilities": domain_probs,
        }

# Hardcoded label mappings for inference (update as needed)
id2label = {0: 'Agriculture', 1: 'Finance', 2: 'General', 3: 'Health'}
label2id = {v: k for k, v in id2label.items()}

def print_prediction(result):
    print(f"\n{'='*70}")
    print(f"Text: {result['text']}")
    print(f"{'─'*70}")
    print(f"  Domain Probabilities:")
    for domain, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"    {domain:<15} {prob:.4f}  {bar}")
    print(f"{'─'*70}")
    print(f"  Model Prediction : {result['predicted_domain']} ({result['confidence']:.4f})")
    if result["thresholded"]:
        print(f"  ⚠ Confidence below threshold → Final Domain: {result['final_domain']}")
    else:
        print(f"  ✓ Final Domain   : {result['final_domain']}")
    print(f"{'='*70}")

"""
Inference Module.

Memuat model final, tokenizer, dan threshold,
lalu menghasilkan prediksi yang konsisten dengan pipeline eksperimen.
"""

import os
import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    LABEL_COLUMNS,
    DEFAULT_THRESHOLD,
    OUTPUT_DIR,
    NUM_LABELS,
)


class EmotionPredictor:
    """
    Inference pipeline untuk multi-label emotion classification.
    
    Memuat model final dan menghasilkan prediksi:
        Text → Tokenizer → Model → 7 Logits → Sigmoid → Threshold → Labels
    """
    
    def __init__(self, model_path=None, threshold=None):
        """
        Args:
            model_path: Path ke saved model. Default: outputs/final_model/
            threshold: Decision threshold. Default dari config.
        """
        if model_path is None:
            model_path = os.path.join(OUTPUT_DIR, "final_model")
        
        self.threshold = threshold or DEFAULT_THRESHOLD
        self.label_columns = LABEL_COLUMNS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model dan tokenizer
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load saved config jika ada
        config_path = os.path.join(model_path, "experiment_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
            self.threshold = self.config.get("threshold", self.threshold)
            print(f"Loaded saved threshold: {self.threshold}")
        
        print(f"Model loaded. Device: {self.device}. Threshold: {self.threshold}")
    
    def predict(self, text):
        """
        Prediksi emosi untuk satu teks.
        
        Args:
            text: String teks komentar
            
        Returns:
            dict: {
                "probabilities": {label: prob, ...},
                "predicted_labels": [label1, label2, ...],
                "threshold": float
            }
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Forward pass (no gradient needed for inference)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Sigmoid activation
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Apply threshold
        predictions = (probs >= self.threshold).astype(int)
        
        # Format results
        probabilities = {
            label: float(round(prob, 4))
            for label, prob in zip(self.label_columns, probs)
        }
        
        predicted_labels = [
            label for label, pred in zip(self.label_columns, predictions)
            if pred == 1
        ]
        
        return {
            "probabilities": probabilities,
            "predicted_labels": predicted_labels,
            "threshold": self.threshold,
        }
    
    def predict_batch(self, texts):
        """
        Prediksi emosi untuk batch teks.
        
        Args:
            texts: List of strings
            
        Returns:
            list[dict]: List of prediction results
        """
        return [self.predict(text) for text in texts]


def save_experiment_config(model_path, config_dict):
    """
    Simpan konfigurasi eksperimen bersama model.
    
    Args:
        model_path: Path ke saved model
        config_dict: Dict dengan konfigurasi (threshold, seed, LR, dll)
    """
    config_path = os.path.join(model_path, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Experiment config saved to: {config_path}")

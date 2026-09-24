"""
Model Module.

Definisi model untuk Multi-label Emotion Classification.
Menggunakan pretrained English Transformer dengan classification head
yang menghasilkan 7 logits (satu per emosi).
"""

from transformers import AutoModelForSequenceClassification
from src.config import MODEL_NAME, NUM_LABELS


def create_model(model_name=None, num_labels=None):
    """
    Load pretrained Transformer dan konfigurasi classification head.
    
    Architecture:
        Pretrained Transformer → Classification Head → 7 Logits
    
    Menggunakan `problem_type="multi_label_classification"` agar
    HuggingFace Trainer secara otomatis menggunakan BCEWithLogitsLoss.
    
    Args:
        model_name: Nama pretrained model. Default dari config.
        num_labels: Jumlah label output. Default dari config.
        
    Returns:
        transformers.PreTrainedModel: Model siap fine-tune.
    """
    if model_name is None:
        model_name = MODEL_NAME
    if num_labels is None:
        num_labels = NUM_LABELS
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
    
    return model


def get_model_info(model):
    """
    Print informasi model untuk dokumentasi.
    
    Args:
        model: HuggingFace model
        
    Returns:
        dict: Informasi model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "model_name": model.config._name_or_path,
        "num_labels": model.config.num_labels,
        "problem_type": model.config.problem_type,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_params_millions": round(total_params / 1e6, 2),
    }
    
    print(f"\n{'='*50}")
    print(f"Model Information")
    print(f"{'='*50}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    return info

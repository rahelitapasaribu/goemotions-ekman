"""
Training Module.

Training loop menggunakan HuggingFace Trainer.
Optimizer, epoch, checkpoint, dan logging terpusat di sini.
"""

import os
import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score

from src.config import (
    SEED,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    CHECKPOINT_DIR,
    LOG_DIR,
    OUTPUT_DIR,
    LABEL_COLUMNS,
)


def compute_metrics_for_trainer(eval_pred):
    """
    Compute metrics callback untuk HuggingFace Trainer.
    
    Trainer memanggil fungsi ini di akhir setiap epoch pada validation set.
    Menggunakan sigmoid karena logits belum diaktivasi.
    
    Args:
        eval_pred: EvalPrediction(predictions, label_ids)
        
    Returns:
        dict: metrics
    """
    predictions, labels = eval_pred
    
    # Sigmoid activation (model menghasilkan raw logits)
    probs = torch.sigmoid(torch.tensor(predictions)).numpy()
    
    # Default threshold 0.5 untuk monitoring selama training
    preds = (probs >= 0.5).astype(int)
    
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    
    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def get_training_args(
    output_dir=None,
    learning_rate=None,
    batch_size=None,
    num_epochs=None,
    weight_decay=None,
    warmup_ratio=None,
    seed=None,
    experiment_name="default",
):
    """
    Buat TrainingArguments dengan konfigurasi lengkap.
    
    Args:
        Semua parameter opsional, default dari config.py.
        experiment_name: Nama eksperimen untuk logging.
        
    Returns:
        TrainingArguments
    """
    return TrainingArguments(
        output_dir=output_dir or os.path.join(CHECKPOINT_DIR, experiment_name),
        
        # Training hyperparameters
        learning_rate=learning_rate or LEARNING_RATE,
        per_device_train_batch_size=batch_size or BATCH_SIZE,
        per_device_eval_batch_size=(batch_size or BATCH_SIZE) * 2,
        num_train_epochs=num_epochs or NUM_EPOCHS,
        weight_decay=weight_decay or WEIGHT_DECAY,
        warmup_ratio=warmup_ratio or WARMUP_RATIO,
        
        # Evaluation & Logging
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir=LOG_DIR,
        
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        
        # Reproducibility
        seed=seed or SEED,
        data_seed=seed or SEED,
        
        # Misc
        report_to="none",  # Disable wandb/tensorboard
        save_total_limit=2,  # Keep only 2 best checkpoints
    )


def train_model(model, train_dataset, val_dataset, training_args=None, experiment_name="default"):
    """
    Fine-tune model menggunakan HuggingFace Trainer.
    
    Training flow:
        Mini-batch → Forward Pass → Logits → BCEWithLogitsLoss → 
        Backpropagation → Gradient → AdamW → Parameter Update
    
    Args:
        model: Pretrained model dari model.py
        train_dataset: EmotionDataset untuk training
        val_dataset: EmotionDataset untuk validation
        training_args: TrainingArguments. Default dibuat dari config.
        experiment_name: Nama eksperimen
        
    Returns:
        tuple: (trainer, train_result)
    """
    if training_args is None:
        training_args = get_training_args(experiment_name=experiment_name)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_trainer,
    )
    
    print(f"\n{'='*50}")
    print(f"Starting training: {experiment_name}")
    print(f"{'='*50}")
    print(f"  Model:         {model.config._name_or_path}")
    print(f"  Learning Rate: {training_args.learning_rate}")
    print(f"  Batch Size:    {training_args.per_device_train_batch_size}")
    print(f"  Epochs:        {training_args.num_train_epochs}")
    print(f"  Seed:          {training_args.seed}")
    
    train_result = trainer.train()
    
    return trainer, train_result


def save_model(trainer, save_path=None):
    """
    Simpan model final dan tokenizer.
    
    Args:
        trainer: HuggingFace Trainer setelah training
        save_path: Path untuk menyimpan. Default dari config.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "final_model")
    
    trainer.save_model(save_path)
    print(f"\nModel saved to: {save_path}")
    
    return save_path

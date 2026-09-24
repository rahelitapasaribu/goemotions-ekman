"""
Main Entry Point.

Orchestrates the full pipeline:
1. Load & validate data
2. Preprocessing
3. Training
4. Evaluation & threshold selection
5. Final test evaluation
6. Save model
"""

import sys
import os
import json
import torch
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    SEED,
    MODEL_NAME,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    MAX_SEQ_LENGTH,
    WEIGHT_DECAY,
    OUTPUT_DIR,
    LABEL_COLUMNS,
)
from src.data_loader import load_goemotions, validate_dataset, get_splits_as_dataframe, print_validation_report
from src.preprocessing import get_tokenizer, create_datasets
from src.model import create_model, get_model_info
from src.train import train_model, save_model, get_training_args
from src.evaluation import (
    compute_metrics_at_threshold,
    find_best_threshold,
    print_full_evaluation,
)
from src.inference import save_experiment_config


def set_seed(seed):
    """Set random seed untuk reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_predictions(trainer, dataset):
    """Extract predictions (probabilities) dari trainer."""
    output = trainer.predict(dataset)
    logits = output.predictions
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = output.label_ids
    return probs, labels


def main():
    print("=" * 60)
    print("GoEmotions-Ekman: Multi-label Emotion Classification")
    print("=" * 60)
    
    # 0. Set seed
    set_seed(SEED)
    print(f"\nRandom seed: {SEED}")
    
    # 1. Load & validate dataset
    print("\n--- Step 1: Load & Validate Dataset ---")
    ds = load_goemotions()
    validation_results = validate_dataset(ds)
    print_validation_report(validation_results)
    
    # 2. Prepare data
    print("\n--- Step 2: Prepare Data ---")
    dataframes = get_splits_as_dataframe(ds)
    tokenizer = get_tokenizer()
    datasets = create_datasets(dataframes, tokenizer)
    
    for split, dataset in datasets.items():
        print(f"  {split}: {len(dataset)} samples")
    
    # 3. Create model
    print("\n--- Step 3: Create Model ---")
    model = create_model()
    get_model_info(model)
    
    # 4. Train
    print("\n--- Step 4: Training ---")
    training_args = get_training_args(experiment_name="main")
    trainer, train_result = train_model(
        model,
        train_dataset=datasets["train"],
        val_dataset=datasets["validation"],
        training_args=training_args,
    )
    
    # 5. Threshold selection on validation set
    print("\n--- Step 5: Threshold Selection (Validation Set) ---")
    val_probs, val_labels = get_predictions(trainer, datasets["validation"])
    best_threshold, threshold_results = find_best_threshold(val_labels, val_probs)
    print(f"\nBest threshold: {best_threshold}")
    
    # 6. Final test evaluation (ONE TIME ONLY)
    print("\n--- Step 6: Final Test Evaluation (FROZEN) ---")
    print(f"  Model: best checkpoint from training")
    print(f"  Threshold: {best_threshold} (from validation)")
    test_probs, test_labels = get_predictions(trainer, datasets["test"])
    test_metrics = print_full_evaluation(test_labels, test_probs, best_threshold, split_name="test")
    
    # 7. Save model & config
    print("\n--- Step 7: Save Model & Config ---")
    model_path = save_model(trainer)
    tokenizer.save_pretrained(model_path)
    
    experiment_config = {
        "seed": SEED,
        "model_name": MODEL_NAME,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "weight_decay": WEIGHT_DECAY,
        "threshold": best_threshold,
        "test_metrics": test_metrics,
    }
    save_experiment_config(model_path, experiment_config)
    
    print("\n" + "=" * 60)
    print("DONE! Model and config saved.")
    print(f"Model path: {model_path}")
    print(f"Best threshold: {best_threshold}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Micro-F1: {test_metrics['micro_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

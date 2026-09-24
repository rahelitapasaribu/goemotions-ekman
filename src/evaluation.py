"""
Evaluation Module.

Metrics, per-label results, threshold evaluation,
confusion/error summaries untuk multi-label classification.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    classification_report,
    multilabel_confusion_matrix,
)

from src.config import LABEL_COLUMNS, THRESHOLD_CANDIDATES, DEFAULT_THRESHOLD


def compute_metrics_at_threshold(y_true, y_prob, threshold=None):
    """
    Hitung semua metrics pada threshold tertentu.
    
    Args:
        y_true: Ground truth multi-hot (n_samples, 7), numpy array
        y_prob: Predicted probabilities (n_samples, 7), numpy array
        threshold: Decision threshold. Default dari config.
        
    Returns:
        dict: Semua metrics
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "threshold": threshold,
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }
    
    # Per-label F1
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, label in enumerate(LABEL_COLUMNS):
        metrics[f"f1_{label}"] = per_label_f1[i]
    
    return metrics


def find_best_threshold(y_true, y_prob, candidates=None, metric="macro_f1"):
    """
    Cari threshold terbaik dari kandidat pada validation set.
    
    Prosedur:
    1. Mulai dari baseline 0.5
    2. Coba semua threshold candidates
    3. Pilih berdasarkan metric yang dipilih (default: Macro-F1)
    
    Args:
        y_true: Ground truth multi-hot
        y_prob: Predicted probabilities
        candidates: List threshold candidates. Default dari config.
        metric: Metric untuk optimisasi. Default "macro_f1".
        
    Returns:
        tuple: (best_threshold, results_per_threshold)
    """
    if candidates is None:
        candidates = THRESHOLD_CANDIDATES
    
    results = []
    for t in candidates:
        m = compute_metrics_at_threshold(y_true, y_prob, threshold=t)
        results.append(m)
    
    best_idx = np.argmax([r[metric] for r in results])
    best_threshold = candidates[best_idx]
    
    print(f"\nThreshold Selection Results (optimize: {metric}):")
    print(f"{'Threshold':<12} {'Micro-F1':<12} {'Macro-F1':<12} {'Hamming':<12}")
    print("-" * 48)
    for r in results:
        marker = " ◄ BEST" if r["threshold"] == best_threshold else ""
        print(f"{r['threshold']:<12.2f} {r['micro_f1']:<12.4f} {r['macro_f1']:<12.4f} {r['hamming_loss']:<12.4f}{marker}")
    
    return best_threshold, results


def print_full_evaluation(y_true, y_prob, threshold, split_name="test"):
    """
    Print laporan evaluasi lengkap untuk satu split.
    
    Args:
        y_true: Ground truth multi-hot
        y_prob: Predicted probabilities
        threshold: Decision threshold
        split_name: Nama split (untuk display)
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = compute_metrics_at_threshold(y_true, y_prob, threshold)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT — {split_name.upper()} SET (threshold={threshold})")
    print(f"{'='*60}")
    
    print(f"\n--- Aggregate Metrics ---")
    print(f"  Micro Precision: {metrics['micro_precision']:.4f}")
    print(f"  Micro Recall:    {metrics['micro_recall']:.4f}")
    print(f"  Micro F1:        {metrics['micro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Hamming Loss:    {metrics['hamming_loss']:.4f}")
    
    print(f"\n--- Per-label F1 ---")
    for label in LABEL_COLUMNS:
        print(f"  {label:<12}: {metrics[f'f1_{label}']:.4f}")
    
    print(f"\n--- Classification Report ---")
    report = classification_report(
        y_true, y_pred,
        target_names=LABEL_COLUMNS,
        zero_division=0
    )
    print(report)
    
    return metrics


def error_analysis(y_true, y_prob, texts, threshold, n_examples=5):
    """
    Analisis error: false positives, false negatives, near-threshold predictions.
    
    Args:
        y_true: Ground truth multi-hot
        y_prob: Predicted probabilities
        texts: List teks asli
        threshold: Decision threshold
        n_examples: Jumlah contoh per kategori error
        
    Returns:
        dict: Error analysis results
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    analysis = {}
    
    # Per-label error counts
    for i, label in enumerate(LABEL_COLUMNS):
        fp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())
        tp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum())
        tn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 0)).sum())
        
        analysis[label] = {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        }
    
    # Near-threshold predictions (probabilities within ±0.05 of threshold)
    near_threshold_mask = np.any(
        np.abs(y_prob - threshold) < 0.05, axis=1
    )
    analysis["near_threshold_count"] = int(near_threshold_mask.sum())
    analysis["near_threshold_ratio"] = float(near_threshold_mask.mean())
    
    # Partial match analysis (some but not all gold labels predicted)
    gold_label_counts = y_true.sum(axis=1)
    pred_label_counts = y_pred.sum(axis=1)
    multi_label_mask = gold_label_counts > 1
    
    if multi_label_mask.sum() > 0:
        correct_per_sample = ((y_pred == 1) & (y_true == 1)).sum(axis=1)
        partial_match = (
            (correct_per_sample > 0) & 
            (correct_per_sample < gold_label_counts) & 
            multi_label_mask
        )
        analysis["partial_match_count"] = int(partial_match.sum())
    
    return analysis

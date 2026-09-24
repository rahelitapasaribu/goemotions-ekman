"""
Configuration module for GoEmotions-Ekman Multi-label Emotion Classification.

Menyimpan semua hyperparameter, paths, dan konfigurasi eksperimen
agar reproducible dan terdokumentasi.
"""

import os

# ============================================================
# Reproducibility
# ============================================================
SEED = 42

# ============================================================
# Dataset
# ============================================================
DATASET_NAME = "AiLab-IMCS-UL/go_emotions-en"
LABEL_COLUMNS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
NUM_LABELS = len(LABEL_COLUMNS)
TEXT_COLUMN = "text"

# ============================================================
# Model
# ============================================================
MODEL_NAME = "bert-base-uncased"

# ============================================================
# Tokenizer / Input Representation
# ============================================================
MAX_SEQ_LENGTH = 128  # Akan di-update berdasarkan hasil EDA

# ============================================================
# Training Hyperparameters
# ============================================================
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# ============================================================
# Threshold
# ============================================================
DEFAULT_THRESHOLD = 0.5
THRESHOLD_CANDIDATES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Buat direktori jika belum ada
for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

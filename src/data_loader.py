"""
Data Loader Module.

Memuat dataset GoEmotions-Ekman dari HuggingFace
dan menyiapkan target labels_ekman menjadi multi-hot.
"""

from datasets import load_dataset
import pandas as pd
import numpy as np

from src.config import DATASET_NAME


EKMAN_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral",
]


def load_goemotions():
    """
    Load dataset dari HuggingFace.
    """
    ds = load_dataset(DATASET_NAME)
    return ds


def labels_to_multihot(label_ids, num_labels=7):
    """
    Mengubah list indeks labels_ekman menjadi multi-hot vector.

    Contoh:
    [3, 5] -> [0, 0, 0, 1, 0, 1, 0]
    """
    vector = np.zeros(num_labels, dtype=np.float32)

    for label_id in label_ids:
        vector[int(label_id)] = 1.0

    return vector


def prepare_dataframe(split):
    """
    Mengubah HuggingFace Dataset menjadi DataFrame dan
    membuat 7 kolom target Ekman.
    """
    df = split.to_pandas()

    # Validasi kolom penting
    required_columns = ["text", "labels_ekman"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan pada dataset.")

    # Ubah labels_ekman menjadi multi-hot
    multihot = np.stack(
        df["labels_ekman"].apply(labels_to_multihot)
    )

    # Buat kolom anger, disgust, ..., neutral
    for i, label in enumerate(EKMAN_LABELS):
        df[label] = multihot[:, i]

    return df


def get_splits_as_dataframe(ds):
    """
    Prepare train, validation, dan test.
    """
    return {
        split_name: prepare_dataframe(ds[split_name])
        for split_name in ds.keys()
    }
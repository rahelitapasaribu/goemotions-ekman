"""
Preprocessing Module.

Handles target encoding (multi-hot vector), tokenizer preparation,
dan pembuatan PyTorch Dataset untuk training/evaluation.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np

from src.config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    LABEL_COLUMNS,
    TEXT_COLUMN,
)


def get_tokenizer(model_name=None):
    """
    Load tokenizer yang konsisten dengan pretrained model.
    
    PENTING: Tidak ada stemming, stop-word removal, atau pembuangan tanda baca.
    Pretrained Transformer sudah mempelajari representasi dari teks asli saat pretraining.
    
    Args:
        model_name: Nama model HuggingFace. Default dari config.
        
    Returns:
        transformers.PreTrainedTokenizer
    """
    if model_name is None:
        model_name = MODEL_NAME
    return AutoTokenizer.from_pretrained(model_name)


class EmotionDataset(Dataset):
    """
    PyTorch Dataset untuk GoEmotions-Ekman.
    
    Setiap item menghasilkan:
    - input_ids: Token IDs dari tokenizer
    - attention_mask: Mask untuk padding tokens
    - labels: Multi-hot vector 7 dimensi (float32 untuk BCEWithLogitsLoss)
    """
    
    def __init__(self, dataframe, tokenizer, max_length=None, label_columns=None):
        """
        Args:
            dataframe: pandas DataFrame dengan kolom teks dan label
            tokenizer: HuggingFace tokenizer
            max_length: Max sequence length. Default dari config.
            label_columns: List nama kolom label. Default dari config.
        """
        self.texts = dataframe[TEXT_COLUMN].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length or MAX_SEQ_LENGTH
        self.label_columns = label_columns or LABEL_COLUMNS
        
        # Multi-hot encoding: setiap label adalah 0 atau 1
        self.labels = dataframe[self.label_columns].values.astype(np.float32)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize: subword tokens → token IDs + attention mask
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def create_datasets(dataframes, tokenizer, max_length=None):
    """
    Buat EmotionDataset untuk setiap split.
    
    Args:
        dataframes: dict[str, pd.DataFrame] dari get_splits_as_dataframe()
        tokenizer: HuggingFace tokenizer
        max_length: Max sequence length
        
    Returns:
        dict[str, EmotionDataset]: Mapping split_name -> EmotionDataset
    """
    datasets = {}
    for split_name, df in dataframes.items():
        datasets[split_name] = EmotionDataset(df, tokenizer, max_length)
    return datasets

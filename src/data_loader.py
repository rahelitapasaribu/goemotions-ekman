"""
Data Loader Module.

Memuat dataset GoEmotions-Ekman dari HuggingFace,
memeriksa schema, dan menyediakan data untuk pipeline.
"""

from datasets import load_dataset
import pandas as pd
import numpy as np

from src.config import (
    DATASET_NAME,
    LABEL_COLUMNS,
    TEXT_COLUMN,
    SEED,
)


def load_goemotions():
    """
    Memuat dataset GoEmotions-Ekman dari HuggingFace.
    
    Returns:
        datasets.DatasetDict: Dataset dengan splits train, validation, test.
    """
    ds = load_dataset(DATASET_NAME)
    return ds


def validate_dataset(ds):
    """
    Memeriksa schema dan kualitas data.
    
    Checks:
    - Keberadaan kolom teks dan label
    - Missing/empty text
    - Label validity (harus 0 atau 1)
    - Duplikasi
    
    Args:
        ds: HuggingFace DatasetDict
        
    Returns:
        dict: Hasil validasi
    """
    results = {}
    
    for split_name in ds:
        split = ds[split_name]
        df = split.to_pandas()
        
        split_results = {
            "total_rows": len(df),
            "columns": list(df.columns),
        }
        
        # Cek kolom yang diharapkan ada
        expected_cols = [TEXT_COLUMN] + LABEL_COLUMNS
        missing_cols = [c for c in expected_cols if c not in df.columns]
        split_results["missing_columns"] = missing_cols
        
        # Cek missing/empty text
        if TEXT_COLUMN in df.columns:
            split_results["null_text"] = int(df[TEXT_COLUMN].isnull().sum())
            split_results["empty_text"] = int((df[TEXT_COLUMN] == "").sum())
        
        # Cek label validity (harus 0 atau 1)
        label_issues = {}
        for col in LABEL_COLUMNS:
            if col in df.columns:
                unique_vals = df[col].unique()
                invalid = [v for v in unique_vals if v not in [0, 1]]
                if invalid:
                    label_issues[col] = invalid
        split_results["invalid_labels"] = label_issues
        
        # Cek duplikasi teks
        if TEXT_COLUMN in df.columns:
            split_results["duplicate_texts"] = int(df[TEXT_COLUMN].duplicated().sum())
        
        results[split_name] = split_results
    
    return results


def get_splits_as_dataframe(ds):
    """
    Konversi setiap split menjadi pandas DataFrame.
    
    Args:
        ds: HuggingFace DatasetDict
        
    Returns:
        dict[str, pd.DataFrame]: Mapping split_name -> DataFrame
    """
    return {split: ds[split].to_pandas() for split in ds}


def print_validation_report(validation_results):
    """
    Print laporan validasi yang readable.
    
    Args:
        validation_results: Output dari validate_dataset()
    """
    for split_name, results in validation_results.items():
        print(f"\n{'='*50}")
        print(f"Split: {split_name}")
        print(f"{'='*50}")
        print(f"  Total rows:      {results['total_rows']}")
        print(f"  Columns:         {results['columns']}")
        print(f"  Missing columns: {results['missing_columns']}")
        
        if "null_text" in results:
            print(f"  Null text:       {results['null_text']}")
            print(f"  Empty text:      {results['empty_text']}")
        
        if results.get("invalid_labels"):
            print(f"  Invalid labels:  {results['invalid_labels']}")
        else:
            print(f"  Invalid labels:  None (all valid)")
        
        if "duplicate_texts" in results:
            print(f"  Duplicate texts: {results['duplicate_texts']}")

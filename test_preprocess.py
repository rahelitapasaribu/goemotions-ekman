from src.data_loader import (
    load_goemotions,
    get_splits_as_dataframe,
)

from src.preprocessing import (
    EmotionDataset,
    get_tokenizer,
)


# =========================
# LOAD DATA
# =========================

ds = load_goemotions()
dataframes = get_splits_as_dataframe(ds)

train_df = dataframes["train"]


# =========================
# TOKENIZER
# =========================

tokenizer = get_tokenizer()


# =========================
# CREATE DATASET
# =========================

train_dataset = EmotionDataset(
    dataframe=train_df,
    tokenizer=tokenizer,
)


# =========================
# CHECK ONE SAMPLE
# =========================

sample = train_dataset[0]

print("Text:")
print(train_df.iloc[0]["text"])

print("\nOriginal Ekman labels:")
print(train_df.iloc[0]["labels_ekman"])

print("\nInput IDs:")
print(sample["input_ids"])

print("\nAttention Mask:")
print(sample["attention_mask"])

print("\nLabels:")
print(sample["labels"])


# =========================
# CHECK SHAPES
# =========================

print("\n--- SHAPES ---")
print("input_ids:", sample["input_ids"].shape)
print("attention_mask:", sample["attention_mask"].shape)
print("labels:", sample["labels"].shape)
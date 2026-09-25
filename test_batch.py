from torch.utils.data import DataLoader

from src.data_loader import (
    load_goemotions,
    get_splits_as_dataframe,
)

from src.preprocessing import (
    EmotionDataset,
    get_tokenizer,
)

from src.config import BATCH_SIZE


# 1. Load data
ds = load_goemotions()
dataframes = get_splits_as_dataframe(ds)

train_df = dataframes["train"]


# 2. Tokenizer
tokenizer = get_tokenizer()


# 3. PyTorch Dataset
train_dataset = EmotionDataset(
    dataframe=train_df,
    tokenizer=tokenizer,
)


# 4. DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# 5. Ambil satu batch
batch = next(iter(train_loader))


# 6. Cek hasil
print("===== BATCH SHAPES =====")

print("input_ids:")
print(batch["input_ids"].shape)

print("\nattention_mask:")
print(batch["attention_mask"].shape)

print("\nlabels:")
print(batch["labels"].shape)


print("\n===== FIRST SAMPLE LABEL =====")
print(batch["labels"][0])


print("\n===== DATA TYPES =====")
print("input_ids:", batch["input_ids"].dtype)
print("attention_mask:", batch["attention_mask"].dtype)
print("labels:", batch["labels"].dtype)
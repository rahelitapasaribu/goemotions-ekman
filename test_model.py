import torch
from torch.utils.data import DataLoader

from src.data_loader import (
    load_goemotions,
    get_splits_as_dataframe,
)

from src.preprocessing import (
    EmotionDataset,
    get_tokenizer,
)

from src.model import create_model
from src.config import BATCH_SIZE


# =========================
# 1. LOAD DATA
# =========================

ds = load_goemotions()
dataframes = get_splits_as_dataframe(ds)

train_df = dataframes["train"]


# =========================
# 2. TOKENIZER & DATASET
# =========================

tokenizer = get_tokenizer()

train_dataset = EmotionDataset(
    dataframe=train_df,
    tokenizer=tokenizer,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# =========================
# 3. GET ONE BATCH
# =========================

batch = next(iter(train_loader))


# =========================
# 4. CREATE MODEL
# =========================

model = create_model()

model.eval()


# =========================
# 5. FORWARD PASS
# =========================

with torch.no_grad():

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )


# =========================
# 6. CHECK OUTPUT
# =========================

print("===== MODEL OUTPUT =====")

print("Logits shape:")
print(outputs.logits.shape)

print("\nLoss:")
print(outputs.loss.item())

print("\nFirst sample logits:")
print(outputs.logits[0])

print("\nFirst sample probabilities:")
print(torch.sigmoid(outputs.logits[0]))
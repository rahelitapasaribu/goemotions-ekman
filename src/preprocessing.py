import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config import MODEL_NAME, MAX_SEQ_LENGTH


LABEL_COLUMNS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral",
]


class EmotionDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=MAX_SEQ_LENGTH,
    ):
        self.texts = dataframe["text"].astype(str).tolist()

        self.labels = dataframe[LABEL_COLUMNS].values.astype("float32")

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.float32
            ),
        }


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer
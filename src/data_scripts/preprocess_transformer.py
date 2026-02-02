import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DDIDataset(Dataset):
    def __init__(
        self,
        csv_path,
        tokenizer_name="distilbert-base-uncased",
        max_length=128
    ):
        self.data = pd.read_csv(csv_path)
        self.texts = self.data["text"].astype(str).tolist()
        self.labels = self.data["label"].tolist()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_dataset(csv_path, max_length=128):
    return DDIDataset(csv_path=csv_path, max_length=max_length)

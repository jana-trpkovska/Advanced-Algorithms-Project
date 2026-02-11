import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DDIDataset(Dataset):
    def __init__(self, csv_path, max_length=128):
        self.df = pd.read_csv(csv_path)
        self.texts = self.df["text"].tolist()
        self.labels = self.df["label"].tolist()
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            use_fast=True
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

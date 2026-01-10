from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import pickle


base_dir = Path(__file__).resolve().parents[2]

DATASETS_DIR = base_dir / "data" / "datasets"
TOKENIZED_DIR = DATASETS_DIR / "tokenized"
TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128


def tokenize_and_save(csv_file, tokenizer, split_name):
    print(f"Processing {csv_file} ...")
    df = pd.read_csv(csv_file)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="np"
    )

    np.save(TOKENIZED_DIR / f"{split_name}_input_ids.npy", encodings["input_ids"])
    np.save(TOKENIZED_DIR / f"{split_name}_attention.npy", encodings["attention_mask"])
    np.save(TOKENIZED_DIR / f"{split_name}_labels.npy", np.array(labels))

    print(f"Saved tokenized {split_name} data.")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with open(TOKENIZED_DIR / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {TOKENIZED_DIR / 'tokenizer.pkl'}")

    for split in ["train", "val", "test"]:
        csv_file = DATASETS_DIR / f"{split}.csv"
        tokenize_and_save(csv_file, tokenizer, split)


if __name__ == "__main__":
    main()

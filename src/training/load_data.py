from pathlib import Path
import numpy as np

base_dir = Path(__file__).resolve().parents[2]
TOKENIZED_DIR = base_dir / "data" / "datasets" / "tokenized"


def load_tokenized_data(split):
    input_ids = np.load(TOKENIZED_DIR / f"{split}_input_ids.npy")
    attention_mask = np.load(TOKENIZED_DIR / f"{split}_attention.npy")
    labels = np.load(TOKENIZED_DIR / f"{split}_labels.npy")

    return input_ids, attention_mask, labels


if __name__ == "__main__":
    train_ids, train_attn, train_labels = load_tokenized_data("train")
    val_ids, val_attn, val_labels = load_tokenized_data("val")
    test_ids, test_attn, test_labels = load_tokenized_data("test")

    print(f"Train: {train_ids.shape}, {train_labels.shape}")
    print(f"Val: {val_ids.shape}, {val_labels.shape}")
    print(f"Test: {test_ids.shape}, {test_labels.shape}")

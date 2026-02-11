from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from src.models.bioclinicalbert import ClinicalBERTClassifier
from src.data_scripts.preprocess_transformer_bioclinicalbert import DDIDataset

MODEL_VERSION = 1
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / f"bioclinicalbert_v{MODEL_VERSION}.pt"

TEST_CSV = BASE_DIR / "test.csv"
PRETRAINED_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

BATCH_SIZE = 16
MAX_LENGTH = 128
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def evaluate():
    print("Loading test dataset...")
    test_dataset = DDIDataset(TEST_CSV, max_length=MAX_LENGTH)
    collator = DataCollatorWithPadding(tokenizer=test_dataset.tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        pin_memory=(DEVICE.type == "cuda")
    )

    print("Loading Bio_ClinicalBERT model...")
    model = ClinicalBERTClassifier(
        pretrained_model_name=PRETRAINED_MODEL_NAME
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_probs = []
    all_labels = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = softmax(logits)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs >= THRESHOLD).astype(int)

    acc = accuracy_score(all_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary", zero_division=0
    )

    print("\nTest Results at Threshold 0.5:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    evaluate()

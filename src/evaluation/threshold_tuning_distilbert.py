from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from src.models.distilbert import DistilBERTClassifier
from src.data_scripts.preprocess_transformer import DDIDataset

MODEL_VERSION = 1
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / f"distilbert_v{MODEL_VERSION}.pt"
TEST_CSV = BASE_DIR / "test.csv"

PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def evaluate_with_thresholds(
    probs,
    y_true,
    thresholds=np.arange(0.30, 0.71, 0.01)
):
    best_f1 = 0.0
    best_threshold = 0.5
    best_metrics = {}

    for t in thresholds:
        y_pred = (probs >= t).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            }

    print("\nBest threshold found")
    print(f"Threshold: {best_threshold:.2f}")
    print("Metrics at best threshold:")
    print(f"Accuracy : {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall   : {best_metrics['recall']:.4f}")
    print(f"F1-score : {best_metrics['f1']:.4f}")


def main():
    print("Loading test dataset...")
    test_dataset = DDIDataset(TEST_CSV, max_length=MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading DistilBERT model...")
    model = DistilBERTClassifier(
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        num_labels=2
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_probs = []
    all_labels = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting probabilities"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = softmax(logits)[:, 1]

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    evaluate_with_thresholds(all_probs, all_labels)


if __name__ == "_main_":
    main()

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from src.models.distilbert import DistilBERTClassifier
from src.data_scripts.preprocess_transformer import DDIDataset

MODEL_VERSION = 5
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / f"distilbert_v{MODEL_VERSION}.pt"

TEST_CSV = BASE_DIR / "test.csv"
PRETRAINED_MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 16
MAX_LENGTH = 96

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def evaluate():
    print("Loading test dataset...")
    test_dataset = DDIDataset(TEST_CSV, max_length=MAX_LENGTH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda")
    )

    print("Loading DistilBERT model...")
    model = DistilBERTClassifier(
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        num_labels=2
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    print("\nTest Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    evaluate()

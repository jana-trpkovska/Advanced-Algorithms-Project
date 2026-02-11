from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, DataCollatorWithPadding
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np

from src.models.bioclinicalbert import ClinicalBERTClassifier
from src.data_scripts.preprocess_transformer_bioclinicalbert import DDIDataset


MODEL_VERSION = 1
BASE_DIR = Path(__file__).resolve().parent

TRAIN_CSV = BASE_DIR / "train.csv"
VAL_CSV = BASE_DIR / "val.csv"
MODEL_PATH = BASE_DIR / f"bioclinicalbert_v{MODEL_VERSION}.pt"

PRETRAINED_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

BATCH_SIZE = 8
MAX_LENGTH = 128
MAX_EPOCHS = 25
PATIENCE = 4
WEIGHT_DECAY = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def train_epoch(model, loader, optimizer, scheduler, loss_fn, epoch):
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=True)

    for batch in progress:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def validate_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    train_dataset = DDIDataset(TRAIN_CSV, max_length=MAX_LENGTH)
    val_dataset = DDIDataset(VAL_CSV, max_length=MAX_LENGTH)

    collator = DataCollatorWithPadding(tokenizer=train_dataset.tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )

    model = ClinicalBERTClassifier(
        pretrained_model_name=PRETRAINED_MODEL_NAME
    ).to(DEVICE)

    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 2e-4},
        ],
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_dataset.labels
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{MAX_EPOCHS} =====")

        if epoch == 3:
            for layer in model.encoder.encoder.layer[-4:]:
                for p in layer.parameters():
                    p.requires_grad = True
            print("Unfroze last 4 encoder layers")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, epoch
        )

        val_loss, val_acc = validate_epoch(
            model, val_loader, loss_fn
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("Best model saved")
        else:
            no_improve += 1
            print(f"No improvement ({no_improve}/{PATIENCE})")

        if no_improve >= PATIENCE:
            print("Early stopping triggered")
            break

    print(f"\nTraining finished. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.models.distilbert import DistilBERTClassifier
from src.data_scripts.preprocess_transformer_distilbert import DDIDataset

MODEL_VERSION = 1
BASE_DIR = Path(__file__).resolve().parent

TRAIN_CSV = BASE_DIR / "train.csv"
VAL_CSV = BASE_DIR / "val.csv"
MODEL_PATH = BASE_DIR / f"distilbert_v{MODEL_VERSION}.pt"

PRETRAINED_MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 8
MAX_LENGTH = 96
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
PATIENCE = 2
MAX_EPOCHS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
torch.set_num_threads(4)


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(dataloader)

    progress_bar = tqdm(
        enumerate(dataloader, start=1),
        total=num_batches,
        desc=f"Epoch {epoch}/{num_epochs}",
        leave=True
    )

    for batch_idx, batch in progress_bar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = correct / total

        progress_bar.set_postfix({
            "batch": f"{batch_idx}/{num_batches}",
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}"
        })

    return total_loss / num_batches


def validate_epoch(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(dataloader)

    with torch.no_grad():
        progress_bar = tqdm(
            enumerate(dataloader, start=1),
            total=num_batches,
            desc="Validation",
            leave=False
        )

        for _, batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = correct / total

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}"
            })

    return total_loss / num_batches, correct / total


def main():
    print("Loading datasets...")
    train_dataset = DDIDataset(TRAIN_CSV, max_length=MAX_LENGTH)
    val_dataset = DDIDataset(VAL_CSV, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing DistilBERT model...")
    model = DistilBERTClassifier(
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        num_labels=2
    ).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("\nStarting training...\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{MAX_EPOCHS} ==========")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            epoch,
            MAX_EPOCHS
        )

        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn)

        print(
            f"Epoch {epoch} Summary | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("Best model saved")
        else:
            epochs_without_improvement += 1
            print(f"No improvement ({epochs_without_improvement}/{PATIENCE})")

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered")
            break

    print(f"\nTraining complete. Best model saved to:\n{MODEL_PATH}")


if __name__ == "__main__":
    main()
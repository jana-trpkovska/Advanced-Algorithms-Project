from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm.keras import TqdmCallback

from src.models.bilstm import create_bilstm_model
from src.training.load_data import load_tokenized_data

base_dir = Path(__file__).resolve().parents[2]

MODEL_DIR = base_dir / "src" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "bilstm_model.h5"


def train():
    X_train, _, y_train = load_tokenized_data("train")
    X_val, _, y_val = load_tokenized_data("val")

    print("Training data shape:", X_train.shape)
    print("Validation data shape:", X_val.shape)

    model = create_bilstm_model()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    )

    tqdm_callback = TqdmCallback(verbose=1)

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            early_stopping,
            checkpoint,
            tqdm_callback
        ]
    )

    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    print("Training model...")
    train()

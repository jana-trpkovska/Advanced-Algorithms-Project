from pathlib import Path
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm.keras import TqdmCallback
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

from src.models.cnn import create_cnn_model
from src.training.load_data import load_tokenized_data

MODEL_VERSION = 7
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "src" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / f"cnn_model_v{MODEL_VERSION}.h5"
EMBEDDINGS_PATH = BASE_DIR / "data" / "embeddings" / "embedding_matrix.npy"


def unfreeze_embeddings(model):
    model.get_layer("embedding").trainable = True
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            "precision",
            "recall",
        ],
    )
    print("Embeddings are now trainable.")
    return model


def compute_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    return dict(zip(classes, weights))


def train():
    print(f"Training CNN model v{MODEL_VERSION}")

    X_train, _, y_train = load_tokenized_data("train")
    X_val, _, y_val = load_tokenized_data("val")

    class_weights = compute_class_weights(y_train)
    print("Class weights:", class_weights)

    embedding_matrix = np.load(EMBEDDINGS_PATH)

    vocab_size, embedding_dim = embedding_matrix.shape
    max_len = X_train.shape[1]

    # Phase 1: frozen embeddings
    model = create_cnn_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix,
        max_len=max_len,
    )

    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[
            early_stopping,
            checkpoint,
            TqdmCallback(verbose=1),
        ]
    )

    # Phase 2: unfrozen embeddings
    model = unfreeze_embeddings(model)

    print("\nPhase 2: Fine-tuning with embeddings trainable")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=8,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, TqdmCallback(verbose=1)],
    )

    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()

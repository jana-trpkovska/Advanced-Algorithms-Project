from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.kmax_pooling_layer_cnn import KMaxPooling

MODEL_VERSION = 9
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "src" / "models" / f"cnn_model_v{MODEL_VERSION}.h5"
TOKENIZED_DIR = BASE_DIR / "data" / "datasets" / "tokenized"


def load_tokenized_data_for_threshold(split):
    input_ids = np.load(TOKENIZED_DIR / f"{split}_input_ids.npy")
    labels = np.load(TOKENIZED_DIR / f"{split}_labels.npy")
    return input_ids, labels


def evaluate_with_thresholds(model, X, y_true, thresholds=np.arange(0.3, 0.71, 0.01)):
    y_probs = model.predict(X, batch_size=32, verbose=1).flatten()

    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    print(f"\nBest threshold: {best_threshold:.2f}")
    print("Metrics at best threshold:")
    print(f"Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall:    {best_metrics['recall']:.4f}")
    print(f"F1 Score:  {best_metrics['f1']:.4f}")


if __name__ == "__main__":
    X_test, y_test = load_tokenized_data_for_threshold("test")
    print(f"Test data shape: {X_test.shape}")

    model = load_model(MODEL_PATH, custom_objects={"KMaxPooling": KMaxPooling})

    evaluate_with_thresholds(model, X_test, y_test)

from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.training.load_data import load_tokenized_data

MODEL_VERSION = 2
base_dir = Path(__file__).resolve().parents[2]
MODEL_PATH = base_dir / "src" / "models" / f"cnn_model_v{MODEL_VERSION}.h5"
BEST_THRESHOLD = 0.5


def evaluate():
    X_test, _, y_test = load_tokenized_data("test")

    print("Test data shape:", X_test.shape)

    model = load_model(MODEL_PATH)

    y_probs = model.predict(X_test, batch_size=32).flatten()
    y_pred = (y_probs >= BEST_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nTest Results with threshold {BEST_THRESHOLD:.2f}:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    evaluate()

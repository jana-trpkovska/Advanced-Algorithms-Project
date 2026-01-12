from pathlib import Path
from tensorflow.keras.models import load_model

from src.training.load_data import load_tokenized_data
from src.models.attention_layer import Attention

MODEL_VERSION = 3
base_dir = Path(__file__).resolve().parents[2]
MODEL_PATH = base_dir / "src" / "models" / f"bilstm_model_v{MODEL_VERSION}.h5"


def evaluate():
    X_test, _, y_test = load_tokenized_data("test")

    print("Test data shape:", X_test.shape)

    model = load_model(MODEL_PATH, custom_objects={"Attention": Attention})

    results = model.evaluate(X_test, y_test, verbose=1)
    loss, accuracy, precision, recall = results

    print("\nTest Results:")
    print(f"Loss:      {loss:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


if __name__ == "__main__":
    evaluate()

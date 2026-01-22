import numpy as np
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
import pickle

from src.models.attention_layer import Attention

EMBEDDING_DIM = 300
MAX_LENGTH = 128
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.3

BASE_DIR = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = BASE_DIR / "data" / "datasets" / "tokenized" / "tokenizer.pkl"
EMBEDDING_MATRIX_PATH = BASE_DIR / "data" / "embeddings" / "embedding_matrix.npy"


def load_embedding_matrix():
    if not EMBEDDING_MATRIX_PATH.exists():
        raise FileNotFoundError(f"Embedding matrix not found at {EMBEDDING_MATRIX_PATH}")
    print(f"Loading embedding matrix from {EMBEDDING_MATRIX_PATH} ...")
    return np.load(EMBEDDING_MATRIX_PATH)


def create_bilstm_model():
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer_obj = pickle.load(f)

    tokenizer_vocab = tokenizer_obj.get_vocab() if hasattr(tokenizer_obj, "get_vocab") else tokenizer_obj.vocab
    vocab_size = len(tokenizer_vocab)

    embedding_matrix = load_embedding_matrix()

    input_ids = Input(shape=(MAX_LENGTH,), name="input_ids")

    x = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        trainable=False,
    )(input_ids)

    x = Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True))(x)
    x = Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=True))(x)
    x = Attention()(x)
    x = LayerNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_ids, outputs=output)

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=3e-4),
        metrics=["accuracy", Precision(), Recall()],
    )

    return model


if __name__ == "__main__":
    model = create_bilstm_model()
    model.summary()

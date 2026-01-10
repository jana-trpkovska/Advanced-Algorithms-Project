from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall

# Hyperparameters
VOCAB_SIZE = 30522        # For bert-base-uncased tokenizer
EMBEDDING_DIM = 128       # size of word embeddings
MAX_LENGTH = 128          # must match the tokenized inputs
LSTM_UNITS = 64
DROPOUT_RATE = 0.3


def create_bilstm_model(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    max_length=MAX_LENGTH,
    lstm_units=LSTM_UNITS,
    dropout_rate=DROPOUT_RATE
):
    input_ids = Input(shape=(max_length,), name="input_ids")
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(input_ids)
    x = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_ids, outputs=output)

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", Precision(), Recall()]
    )

    return model


if __name__ == "__main__":
    model = create_bilstm_model()
    model.summary()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam

# Hyperparameters
VOCAB_SIZE = 30522        # For bert-base-uncased tokenizer
EMBEDDING_DIM = 128       # size of word embeddings
MAX_LENGTH = 128          # must match the tokenized inputs
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.3


def create_bilstm_model():
    input_ids = Input(shape=(MAX_LENGTH,), name="input_ids")
    x = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM)(input_ids)
    x = Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True))(x)
    x = Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=False))(x)
    x = LayerNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_ids, outputs=output)

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=3e-4),
        metrics=["accuracy", Precision(), Recall()]
    )

    return model


if __name__ == "__main__":
    model = create_bilstm_model()
    model.summary()

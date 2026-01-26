from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall


def create_cnn_model(
        vocab_size,
        embedding_dim,
        embedding_matrix,
        max_len,
        num_filters=128,
        kernel_sizes=(3, 4, 5),
        dropout_rate=0.5,
):
    inputs = Input(shape=(max_len,), name="input_ids")

    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False,
        name="embedding",
    )(inputs)

    conv_blocks = []
    for k in kernel_sizes:
        conv1 = Conv1D(
            filters=num_filters,
            kernel_size=k,
            activation="relu",
            padding="same",
            kernel_regularizer=l2(1e-4),
            name=f"conv1_{k}",
        )(embedding)

        conv2 = Conv1D(
            filters=num_filters * 2,
            kernel_size=k,
            activation="relu",
            padding="same",
            kernel_regularizer=l2(1e-4),
            name=f"conv2_{k}",
        )(conv1)

        pooled = GlobalMaxPooling1D(name=f"pool_{k}")(conv2)
        conv_blocks.append(pooled)

    x = Concatenate(name="concat")(conv_blocks)

    x = Dropout(dropout_rate, name="dropout")(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4), name="dense")(x)
    outputs = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )

    return model

from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import tanh, dot, softmax, sum as ksum, concatenate


class MultiHeadAttention(Layer):
    def __init__(self, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_weights = []

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        time_steps = input_shape[1]

        for i in range(self.num_heads):
            self.attention_weights.append(
                self.add_weight(
                    name=f"att_weight_{i}",
                    shape=(hidden_size, 1),
                    initializer="random_normal",
                    trainable=True,
                )
            )

        self.bias = self.add_weight(
            name="att_bias",
            shape=(time_steps, 1),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x):
        contexts = []

        for W in self.attention_weights:
            e = tanh(dot(x, W) + self.bias)
            a = softmax(e, axis=1)
            context = ksum(x * a, axis=1)
            contexts.append(context)

        return concatenate(contexts)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2] * self.num_heads)

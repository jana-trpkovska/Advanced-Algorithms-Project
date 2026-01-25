from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import tanh, dot, softmax, sum as ksum


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
            name="attention_weights"
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
            name="attention_bias"
        )
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        e = tanh(dot(inputs, self.W) + self.b)
        a = softmax(e, axis=1)
        output = ksum(inputs * a, axis=1)
        return output

from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import tanh, dot, softmax, sum as ksum


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tanh(dot(x, self.W) + self.b)
        a = softmax(e, axis=1)
        output = ksum(x * a, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

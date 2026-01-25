from tensorflow.keras.layers import Layer
import tensorflow as tf


class KMaxPooling(Layer):
    def __init__(self, k=3, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        transposed = tf.transpose(inputs, [0, 2, 1])
        top_k = tf.nn.top_k(transposed, k=self.k, sorted=True).values
        flattened = tf.reshape(top_k, [tf.shape(inputs)[0], -1])
        return flattened

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2] * self.k)

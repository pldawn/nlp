import tensorflow as tf
import keras as krs


class SingleVectorAddtiveAttention(krs.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.kernel = None
        super(SingleVectorAddtiveAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[-1], self.units),
                                      initializer='glorot_uniform', trainable=True)

    def call(self, inputs, **kwargs):
        logits = tf.reduce_sum(inputs @ self.kernel, axis=-1)
        scores = krs.activations.softmax(logits)
        weighted_inputs = inputs * tf.expand_dims(scores, -1)
        output = tf.reduce_sum(weighted_inputs, -2)

        return output

    def get_config(self):
        config = {
            "units": self.units
        }

        return config


if __name__ == "__main__":
    agent = SingleVectorAddtiveAttention(2)
    print(agent(tf.constant([[[1.0], [1.0]], [[1.0], [1.0]]])))

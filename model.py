import tensorflow as tf
from tensorflow.keras import layers, Model

class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.W = layers.Dense(64)

    def call(self, inputs):
        score = self.W(inputs)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context, weights

def build_model(window):
    inputs = layers.Input(shape=(window, 1))
    lstm_out = layers.LSTM(64, return_sequences=True)(inputs)
    context, weights = AttentionLayer()(lstm_out)
    outputs = layers.Dense(1)(context)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
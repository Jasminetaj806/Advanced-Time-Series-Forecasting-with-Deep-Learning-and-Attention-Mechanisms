import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from model import AttentionLayer

WINDOW = 50

def create_sequences(data, window=50):
    X = []
    for i in range(len(data)-window):
        X.append(data[i:i+window])
    return np.array(X)

df = pd.read_csv("synthetic_series.csv")
data = df['value'].values

X = create_sequences(data, WINDOW)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = load_model("attention_model.keras",
                   custom_objects={'AttentionLayer': AttentionLayer},
                   compile=False)

lstm_model = tf.keras.Model(inputs=model.input,
                            outputs=model.layers[1].output)

lstm_out = lstm_model.predict(X[:1])

_, attention_weights = model.layers[2](lstm_out)

plt.imshow(attention_weights[0].numpy().reshape(1, -1), aspect="auto", cmap="viridis")
plt.colorbar()
plt.title("Attention Weights Heatmap")
plt.xlabel("Time Steps")
plt.yticks([])
plt.show()
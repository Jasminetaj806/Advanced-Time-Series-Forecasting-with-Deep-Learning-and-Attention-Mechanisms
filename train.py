import pandas as pd
import numpy as np
from model import build_model

WINDOW = 50

def create_sequences(data, window=50):
    X, y = [], []
    for i in range(len(data)-window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

df = pd.read_csv("synthetic_series.csv")
data = df['value'].values

X, y = create_sequences(data, WINDOW)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = build_model(WINDOW)
model.fit(X, y, epochs=5, batch_size=32)
model.save("attention_model.keras")

print("Model trained and saved")
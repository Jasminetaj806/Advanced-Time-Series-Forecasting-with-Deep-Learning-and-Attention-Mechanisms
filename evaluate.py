import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from model import AttentionLayer

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

model = load_model("attention_model.keras",
                   custom_objects={'AttentionLayer': AttentionLayer},
                   compile=False)

pred = model.predict(X).flatten()

mae = np.mean(np.abs(y - pred))
rmse = np.sqrt(np.mean((y - pred)**2))
mape = np.mean(np.abs((y - pred) / y)) * 100

print("Deep Model MAE :", mae)
print("Deep Model RMSE:", rmse)
print("Deep Model MAPE:", mape)
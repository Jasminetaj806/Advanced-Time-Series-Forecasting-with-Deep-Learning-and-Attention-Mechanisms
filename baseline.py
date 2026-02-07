import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

sarima_model = SARIMAX(data[:split+WINDOW], order=(1,1,1), seasonal_order=(1,1,1,50))
sarima_fit = sarima_model.fit(disp=False)
sarima_pred = sarima_fit.predict(start=split+WINDOW, end=len(data)-1)
y_test_sarima = data[split+WINDOW:]

mae_sarima = mean_absolute_error(y_test_sarima, sarima_pred)
rmse_sarima = np.sqrt(mean_squared_error(y_test_sarima, sarima_pred))
mape_sarima = np.mean(np.abs((y_test_sarima - sarima_pred)/y_test_sarima))*100

print("SARIMA MAE :", mae_sarima)
print("SARIMA RMSE:", rmse_sarima)
print("SARIMA MAPE:", mape_sarima)

X_flat = X.reshape(X.shape[0], X.shape[1])
X_train_f, X_test_f = X_flat[:split], X_flat[split:]

mlp = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=300)
mlp.fit(X_train_f, y_train)
pred_mlp = mlp.predict(X_test_f)

mae_mlp = mean_absolute_error(y_test, pred_mlp)
rmse_mlp = np.sqrt(mean_squared_error(y_test, pred_mlp))
mape_mlp = np.mean(np.abs((y_test - pred_mlp)/y_test))*100

print("MLP MAE :", mae_mlp)
print("MLP RMSE:", rmse_mlp)
print("MLP MAPE:", mape_mlp)
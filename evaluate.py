from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

preds = model.predict([X_test, decoder_input_test])
preds = preds.reshape(-1)
y_true = y_test.reshape(-1)

mae = mean_absolute_error(y_true, preds)
rmse = mean_squared_error(y_true, preds, squared=False)
mape = np.mean(np.abs((y_true - preds) / y_true)) * 100

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")
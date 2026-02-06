from sklearn.metrics import mean_absolute_error, mean_squared_error

model.eval()
with torch.no_grad():
    preds, attn = model(torch.tensor(X_test, dtype=torch.float32).to(device))
    preds = preds.cpu().numpy().squeeze()

rmse = np.sqrt(mean_squared_error(y_test.flatten(), preds.flatten()))
mae = mean_absolute_error(y_test.flatten(), preds.flatten())
mape = np.mean(np.abs((y_test.flatten() - preds.flatten()) / y_test.flatten())) * 100

print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}%")

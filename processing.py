from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, lookback=30, horizon=10):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i:i+lookback, :-1])
        y.append(data[i+lookback:i+lookback+horizon, -1])
    return np.array(X), np.array(y)

LOOKBACK = 30
HORIZON = 10

X, y = create_sequences(scaled_data, LOOKBACK, HORIZON)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

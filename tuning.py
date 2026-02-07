configs = [
    {"units": 32, "batch": 16},
    {"units": 64, "batch": 32},
    {"units": 128, "batch": 32}
]

best_mae = float("inf")

for cfg in configs:
    model, _ = build_seq2seq_attention(INPUT_STEPS, OUTPUT_STEPS, FEATURES)
    model.fit(X_train, y_train, epochs=20, batch_size=cfg["batch"], verbose=0)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)

    if mae < best_mae:
        best_mae = mae
        best_cfg = cfg

print("Best config:", best_cfg)
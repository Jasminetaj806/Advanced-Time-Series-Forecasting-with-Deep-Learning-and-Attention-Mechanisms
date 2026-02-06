device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Seq2Seq(
    Encoder(4, 64, 2),
    Decoder(64, 1, Attention(64)),
    HORIZON
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    output, _ = model(X_train_t)
    loss = criterion(output.squeeze(), y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

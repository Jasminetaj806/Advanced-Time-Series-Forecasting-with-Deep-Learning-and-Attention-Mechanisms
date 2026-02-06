class BaselineLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 64, batch_first=True)
        self.fc = nn.Linear(64, HORIZON)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, attention):
        super().__init__()
        self.attention = attention
        self.lstm = nn.LSTM(hidden_dim + 1, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        lstm_input = torch.cat((input.unsqueeze(2), context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell, attn_weights

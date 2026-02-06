import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, hidden_dim, attention):
        super().__init__()
        self.attention = attention
        self.lstm = nn.LSTM(hidden_dim + 1, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: (B, 1)
        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((input.unsqueeze(1), context), dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell, attn_weights
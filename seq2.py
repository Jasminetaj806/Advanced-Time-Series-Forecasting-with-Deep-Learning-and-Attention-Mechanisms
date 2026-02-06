class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, horizon):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.horizon = horizon

    def forward(self, src):
        encoder_outputs, hidden, cell = self.encoder(src)
        input = src[:, -1, 0]  # start token
        outputs, attentions = [], []

        for _ in range(self.horizon):
            output, hidden, cell, attn = self.decoder(
                input.unsqueeze(1), hidden, cell, encoder_outputs
            )
            outputs.append(output)
            attentions.append(attn)
            input = output.squeeze(1)

        return torch.stack(outputs, dim=1), torch.stack(attentions)

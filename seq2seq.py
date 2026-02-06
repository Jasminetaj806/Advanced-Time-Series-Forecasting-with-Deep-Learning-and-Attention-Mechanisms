import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, horizon):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.horizon = horizon

    def forward(self, src, target=None, teacher_forcing=0.5):
        B = src.size(0)
        encoder_outputs, hidden, cell = self.encoder(src)

        decoder_input = src[:, -1, -1]  # last known target value
        outputs = []
        attentions = []

        for t in range(self.horizon):
            out, hidden, cell, attn = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            outputs.append(out)
            attentions.append(attn)

            if target is not None and torch.rand(1).item() < teacher_forcing:
                decoder_input = target[:, t]
            else:
                decoder_input = out.squeeze(1)

        return torch.stack(outputs, dim=1), torch.stack(attentions, dim=1)
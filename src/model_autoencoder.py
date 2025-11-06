# Feed-forward autoencoder for pooled CAN-bus window features.
# I keep the architecture simple and symmetric to establish a fast baseline.

import torch as th
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 64, 32), dropout: float = 0.10):
        super().__init__()
        # Encoder: I progressively reduce dimensionality to a compact latent space.
        enc = []
        d = in_dim
        for h in hidden:
            enc += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.encoder = nn.Sequential(*enc)

        # Decoder: I mirror the encoder back to input dimension.
        dec = []
        for h in list(hidden[::-1])[1:]:
            dec += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        dec += [nn.Linear(d, in_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        # I return both reconstruction and latent so I can inspect embeddings later if needed.
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

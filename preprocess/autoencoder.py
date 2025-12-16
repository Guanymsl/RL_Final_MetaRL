import torch.nn as nn
from environment.param import AE_LATENT_DIM

class AutoEncoder(nn.Module):
    def __init__(self, hidden_dim=AE_LATENT_DIM):
        super(AutoEncoder, self).__init__()
        self.input_dim = 72
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

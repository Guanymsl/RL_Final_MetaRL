import torch.nn as nn

class CardAutoEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super(CardAutoEncoder, self).__init__()
        self.input_channels = 6
        self.input_height = 4
        self.input_width = 13
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.input_height * self.input_width, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64 * self.input_height * self.input_width),
            nn.ReLU(),
            nn.Unflatten(1, (64, self.input_height, self.input_width)),
            nn.ConvTranspose2d(64, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, self.input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

class ActionAutoEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super(ActionAutoEncoder, self).__init__()
        raise NotImplementedError
        self.input_channels = 4 * self.n
        self.input_height = 4
        self.input_width = self.m
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.input_height * self.input_width, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64 * self.input_height * self.input_width),
            nn.ReLU(),
            nn.Unflatten(1, (64, self.input_height, self.input_width)),
            nn.ConvTranspose2d(64, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, self.input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

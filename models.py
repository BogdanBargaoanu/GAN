import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class ConvGenerator(nn.Module):

    def __init__(self, latent_dim=100):
        super().__init__()

        # Project and reshape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True)
        )

        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 7x7 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),  # 14x14 -> 28x28
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 7, 7)
        x = self.conv_blocks(x)
        return x.view(x.size(0), -1)  # Flatten to 784


class ConvDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # 28x28 -> 14x14
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 4, 2, 1),  # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
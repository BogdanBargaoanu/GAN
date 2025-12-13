import torch
import torch.nn as nn
from models import Generator, Discriminator
from data import get_dataloader
from utils import save_generated_images

# Hyperparameters
BATCH_SIZE = 64
LATENT_DIM = 100
EPOCHS = 15
LR = 0.0002

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
dataloader = get_dataloader(BATCH_SIZE)

# Models
G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)

# Loss & Optimizers
criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=LR)
opt_D = torch.optim.Adam(D.parameters(), lr=LR)

for epoch in range(EPOCHS):
    for real, _ in dataloader:
        real = real.view(real.size(0), -1).to(device)
        batch_size = real.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ----- Train Discriminator -----
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake = G(z)

        D_loss = (
            criterion(D(real), real_labels) +
            criterion(D(fake.detach()), fake_labels)
        )

        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        # ----- Train Generator -----
        G_loss = criterion(D(fake), real_labels)

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] | D Loss: {D_loss.item():.4f} | G Loss: {G_loss.item():.4f}")

    # 🔹 Generate & save images
    save_generated_images(G, LATENT_DIM, epoch + 1, device)
import torch
import torch.nn as nn
from models import Generator, Discriminator
from data import get_dataloader
from utils import save_generated_images
import time
import sys
import os

# Hyperparameters
BATCH_SIZE = 64
LATENT_DIM = 100
EPOCHS = 15
LR = 0.0005

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 50)
print(f"Training on: {device}")
print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Latent Dim: {LATENT_DIM}")
print("=" * 50)

# Create output directory
os.makedirs("generated_images", exist_ok=True)

# Data
print("Loading dataset...")
dataloader = get_dataloader(BATCH_SIZE)

# Models
print("Initializing models...")
G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)

# Loss & Optimizers
criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=LR)
opt_D = torch.optim.Adam(D.parameters(), lr=LR)

# Training statistics
total_batches = len(dataloader)
epoch_times = []

print(f"\nStarting training with {total_batches} batches per epoch...")
print("=" * 50)

for epoch in range(EPOCHS):
    epoch_start_time = time.time()

    # Initialize counters
    D_loss_sum = 0
    G_loss_sum = 0
    batch_count = 0

    # Create a progress bar for batches
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.view(real.size(0), -1).to(device)
        batch_size = real.size(0)

        # Create labels
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

        # Update statistics
        D_loss_sum += D_loss.item()
        G_loss_sum += G_loss.item()
        batch_count += 1

        # Update progress bar
        progress = (batch_idx + 1) / total_batches
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        sys.stdout.write(f'\rEpoch [{epoch + 1:02d}/{EPOCHS}] | {bar} | '
                         f'Batch [{batch_idx + 1:03d}/{total_batches:03d}] | '
                         f'D Loss: {D_loss.item():.4f} | G Loss: {G_loss.item():.4f}')
        sys.stdout.flush()

    # Calculate epoch statistics
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # Clear the line and print epoch summary
    sys.stdout.write('\r' + ' ' * 100 + '\r')  # Clear line

    print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] completed in {epoch_time:.1f}s "
          f"(avg: {avg_epoch_time:.1f}s) | "
          f"D Loss: {D_loss_sum / batch_count:.4f} | G Loss: {G_loss_sum / batch_count:.4f}")

    # Generate & save images
    print(f"Saving generated images for epoch {epoch + 1}...")
    save_generated_images(G, LATENT_DIM, epoch + 1, device, save_dir="generated_images")

    # Estimate remaining time
    if len(epoch_times) > 0:
        remaining_epochs = EPOCHS - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        hours = int(estimated_remaining // 3600)
        minutes = int((estimated_remaining % 3600) // 60)
        seconds = int(estimated_remaining % 60)

        if hours > 0:
            print(f"⏰ Estimated time remaining: {hours}h {minutes}m {seconds}s")
        elif minutes > 0:
            print(f"⏰ Estimated time remaining: {minutes}m {seconds}s")
        else:
            print(f"⏰ Estimated time remaining: {seconds}s")

print("\n" + "=" * 50)
print("Training completed!")
print(f"Total training time: {sum(epoch_times):.1f} seconds")
print(f"Average epoch time: {sum(epoch_times) / len(epoch_times):.1f} seconds")
print("=" * 50)

# Save final models
print("Saving final models...")
torch.save(G.state_dict(), "generator_final.pth")
torch.save(D.state_dict(), "discriminator_final.pth")
print("Models saved as 'generator_final.pth' and 'discriminator_final.pth'")
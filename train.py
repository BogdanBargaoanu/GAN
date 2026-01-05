import json

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
EPOCHS = 40
LR_G = 0.0002   # Generator learning rate
LR_D = 0.0004   # Discriminator learning rate
SAVE_INTERVAL = 5  # Save checkpoints every N epochs

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 50)
print(f"Training on: {device}")
print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Latent Dim: {LATENT_DIM}")
print("=" * 50)

# Create output directories
os.makedirs("generated_images", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Data
print("Loading dataset...")
dataloader = get_dataloader(BATCH_SIZE)

# Models
print("Initializing models...")
G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)

# Loss & Optimizers
criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))

# Training statistics
total_batches = len(dataloader)
epoch_times = []
training_history = {
    "d_losses": [],
    "g_losses": [],
    "epoch_times": []
}

# Best model tracking
best_g_loss = float('inf')
best_epoch = 0

fixed_noise = torch.randn(64, LATENT_DIM).to(device)

print(f"\nStarting training with {total_batches} batches per epoch...")
print("=" * 50)

for epoch in range(EPOCHS):
    epoch_start_time = time.time()

    # Initialize counters
    D_loss_sum = 0
    G_loss_sum = 0
    batch_count = 0

    # Set models to training mode
    G.train()
    D.train()

    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.view(real.size(0), -1).to(device)
        batch_size = real.size(0)

        # Create labels with label smoothing for better training
        real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # Label smoothing
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ----- Train Discriminator -----
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake = G(z)

        D_real_loss = criterion(D(real), real_labels)
        D_fake_loss = criterion(D(fake.detach()), fake_labels)
        D_loss = D_real_loss + D_fake_loss

        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        # ----- Train Generator -----
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake = G(z)
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

    avg_d_loss = D_loss_sum / batch_count
    avg_g_loss = G_loss_sum / batch_count

    # Store training history
    training_history["d_losses"].append(avg_d_loss)
    training_history["g_losses"].append(avg_g_loss)
    training_history["epoch_times"].append(epoch_time)

    # Clear the line and print epoch summary
    sys.stdout.write('\r' + ' ' * 120 + '\r')  # Clear line

    print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] completed in {epoch_time:.1f}s "
          f"(avg: {avg_epoch_time:.1f}s) | "
          f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}", end="")

    # Check if this is the best generator so far
    if avg_g_loss < best_g_loss:
        best_g_loss = avg_g_loss
        best_epoch = epoch + 1
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': G.state_dict(),
            'optimizer_state_dict': opt_G.state_dict(),
            'loss': avg_g_loss,
        }, "checkpoints/generator_best.pth")

    print()  # New line

    # Generate & save images using fixed noise
    G.eval()
    with torch.no_grad():
        print(f"Saving generated images for epoch {epoch + 1}...")
        save_generated_images(G, LATENT_DIM, epoch + 1, device, save_dir="generated_images")

    # Save periodic checkpoints
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': G.state_dict(),
            'discriminator_state_dict': D.state_dict(),
            'optimizer_G_state_dict': opt_G.state_dict(),
            'optimizer_D_state_dict': opt_D.state_dict(),
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
        }, f"checkpoints/checkpoint_epoch_{epoch + 1}.pth")
        print(f"Checkpoint saved at epoch {epoch + 1}")

    # Estimate remaining time
    if len(epoch_times) > 0:
        remaining_epochs = EPOCHS - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        hours = int(estimated_remaining // 3600)
        minutes = int((estimated_remaining % 3600) // 60)
        seconds = int(estimated_remaining % 60)

        if hours > 0:
            print(f"Estimated time remaining: {hours}h {minutes}m {seconds}s")
        elif minutes > 0:
            print(f"Estimated time remaining: {minutes}m {seconds}s")
        else:
            print(f"Estimated time remaining: {seconds}s")

    print("-" * 50)

print("\n" + "=" * 50)
print("Training completed!")
print(f"Total training time: {sum(epoch_times) / 60:.1f} minutes")
print(f"Average epoch time: {sum(epoch_times) / len(epoch_times):.1f} seconds")
print(f"Best generator loss: {best_g_loss:.4f} at epoch {best_epoch}")
print("=" * 50)

# Save final models
print("\nSaving final models...")
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': G.state_dict(),
    'optimizer_state_dict': opt_G.state_dict(),
    'loss': avg_g_loss,
}, "generator_final.pth")

torch.save({
    'epoch': EPOCHS,
    'model_state_dict': D.state_dict(),
    'optimizer_state_dict': opt_D.state_dict(),
    'loss': avg_d_loss,
}, "discriminator_final.pth")

# Save training history
with open("training_history.json", "w") as f:
    json.dump(training_history, f, indent=2)

print("Final models saved as 'generator_final.pth' and 'discriminator_final.pth'")
print("Best generator saved as 'checkpoints/generator_best.pth'")
print("Training history saved as 'training_history.json'")
print("\nDone!")
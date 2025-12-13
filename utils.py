import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os

def save_generated_images(generator, latent_dim, epoch, device, n_images=16, save_dir="generated_images"):
    generator.eval()

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(n_images, latent_dim).to(device)
        fake_images = generator(z)
        fake_images = fake_images.view(-1, 1, 28, 28)

        grid = make_grid(fake_images, nrow=4, normalize=True)

        plt.figure(figsize=(4, 4))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.title(f"Epoch {epoch}")
        save_path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()

    generator.train()
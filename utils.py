import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def save_generated_images(generator, latent_dim, epoch, device, n_images=16):
    generator.eval()

    with torch.no_grad():
        z = torch.randn(n_images, latent_dim).to(device)
        fake_images = generator(z)
        fake_images = fake_images.view(-1, 1, 28, 28)

        grid = make_grid(fake_images, nrow=4, normalize=True)

        plt.figure(figsize=(4, 4))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.title(f"Epoch {epoch}")
        plt.savefig(f"generated_epoch_{epoch}.png")
        plt.close()

    generator.train()
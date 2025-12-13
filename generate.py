import sys
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models import Generator

LATENT_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATOR_PATH = "generator_final.pth"

def generate_images(n_images=16):
    # Load generator
    generator = Generator(LATENT_DIM).to(DEVICE)
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    generator.eval()

    with torch.no_grad():
        z = torch.randn(n_images, LATENT_DIM).to(DEVICE)
        fake_images = generator(z)
        fake_images = fake_images.view(-1, 1, 28, 28)

        grid = make_grid(
            fake_images,
            nrow=int(n_images ** 0.5),
            normalize=True
        )

        plt.figure(figsize=(5, 5))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    generate_images(num_images)
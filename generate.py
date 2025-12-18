import sys
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models import Generator

LATENT_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATOR_PATH = "checkpoints/generator_best.pth"  # Updated path


def generate_images(n_images=16):
    # Load generator
    generator = Generator(LATENT_DIM).to(DEVICE)

    checkpoint = torch.load(GENERATOR_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded generator from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Generator loss: {checkpoint.get('loss', 'unknown'):.4f}")
    else:
        generator.load_state_dict(checkpoint)

    generator.eval()

    print(f"Generating {n_images} images...")

    with torch.no_grad():
        z = torch.randn(n_images, LATENT_DIM).to(DEVICE)
        fake_images = generator(z)
        fake_images = fake_images.view(-1, 1, 28, 28)

        grid = make_grid(
            fake_images,
            nrow=int(n_images ** 0.5),
            normalize=True
        )

        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).cpu(), cmap='gray')
        plt.axis("off")
        plt.title(f"Generated Images ({n_images} samples)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    generate_images(num_images)
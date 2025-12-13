# MNIST GAN - Handwritten Digit Generation

A Generative Adversarial Network (GAN) implementation using PyTorch Lightning to generate handwritten digits similar to the MNIST dataset.

## Setup

1. **Clone or create the project directory:**
   ```bash
   mkdir mnist-gan
   cd mnist-gan
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Train the GAN model:
```bash
python train.py
```

The training will:
- Automatically download the MNIST dataset
- Train for 20 epochs (configurable in `train.py`)

### Generate Images

Generate images from a trained model:
```bash

```

Arguments:
- First argument: Path to checkpoint file
- Second argument (optional): Number of images to generate (default: 16)

## Model Architecture

### Generator
- **Input:** Random noise vector (latent_dim=100)
- **Architecture:** Linear → ConvTranspose2d layers → Conv2d
- **Output:** 28x28 grayscale image

### Discriminator
- **Input:** 28x28 grayscale image
- **Architecture:** Conv2d layers → Linear layers
- **Output:** Single value (0=fake, 1=real)

## Configuration

Edit `train.py` to customize:
- `LATENT_DIM`: Size of random noise vector (default: 100)
- `LR`: Learning rate for both networks (default: 0.0005)
- `EPOCHS`: Number of training epochs (default: 20)
- `BATCH_SIZE`: Batch size for training (default: 128)

## How GANs Work

1. **Generator** creates fake images from random noise
2. **Discriminator** tries to distinguish real images from fake ones
3. They compete in a minimax game:
   - Discriminator wants to correctly classify real vs fake
   - Generator wants to fool the discriminator
4. Through this competition, the Generator learns to create realistic images

## Troubleshooting

**GPU not detected:**
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- The code will automatically fall back to CPU if no GPU is available

**Out of memory:**
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `LATENT_DIM` in `config.py`

**Poor quality images:**
- Train for more epochs
- Adjust learning rate
- Try different random seeds

## References

- Original GAN paper: [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- MNIST Dataset: [https://www.kaggle.com/datasets/hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

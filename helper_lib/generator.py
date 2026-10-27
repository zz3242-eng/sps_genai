import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def generate_samples(model, device, num_samples=10):
    generator = model
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, device=device)
        images = generator(noise)

    grid = make_grid(images, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.show()
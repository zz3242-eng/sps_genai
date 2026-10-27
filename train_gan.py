import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_mnist_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples
from helper_lib.utils import save_model

device = (
    torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
)
print(f"Using device: {device}")

train_loader = get_mnist_loader('./data', batch_size=128)

generator, discriminator = get_model("GAN")
generator.to(device)
discriminator.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

trained_generator, trained_discriminator = train_gan(
    (generator, discriminator),
    train_loader,
    criterion,
    (optimizer_g, optimizer_d),
    device=device,
    epochs=20
)

generate_samples(trained_generator, device, num_samples=10)

save_model(trained_generator, 'generator.pth')
print("Generator saved to generator.pth")
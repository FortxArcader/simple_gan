import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Generator Model
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 0.0002
z_dim = 64
image_dim = 28 * 28  # MNIST images are 28x28
batch_size = 128
num_epochs = 50

# Initialize models
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

# Optimizers
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, image_dim).to(device)  # Flatten images
        batch_size = real.size(0)

        # Train Discriminator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        disc_real = disc(real)
        disc_fake = disc(fake)
        
        # Loss for real and fake
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2

        opt_disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # Train Generator
        noise = torch.randn(batch_size, z_dim).to(device)  # Generate new noise for the generator
        fake = gen(noise)  # Generate new fake images

        output = disc(fake)
        lossG = criterion(output, torch.ones_like(output))  # We want the generator to fool the discriminator

        opt_gen.zero_grad()
        lossG.backward()  # No need for retain_graph=True here
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

print("Training complete.")

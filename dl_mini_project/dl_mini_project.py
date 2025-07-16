import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Hyperparameters
input_dim = 784
n_l1 = 1000
n_l2 = 1000
z_dim = 2
batch_size = 100
n_epochs = 50
learning_rate = 0.001
beta1 = 0.9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataLoader
transform = transforms.ToTensor()
mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Networks
torch.manual_seed(0)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, n_l1), nn.ReLU(),
            nn.Linear(n_l1, n_l2), nn.ReLU(),
            nn.Linear(n_l2, z_dim)
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, n_l2), nn.ReLU(),
            nn.Linear(n_l2, n_l1), nn.ReLU(),
            nn.Linear(n_l1, input_dim), nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, n_l1), nn.ReLU(),
            nn.Linear(n_l1, n_l2), nn.ReLU(),
            nn.Linear(n_l2, 1)
        )
    def forward(self, z):
        return self.net(z)

# Initialize models
encoder = Encoder().to(device)
decoder = Decoder().to(device)
discriminator = Discriminator().to(device)

# Losses and optimizers
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()
optimizer_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_g = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Training loop
for epoch in range(1, n_epochs + 1):
    for i, (images, _) in enumerate(dataloader):
        images = images.view(-1, input_dim).to(device)

        # === Train autoencoder ===
        optimizer_ae.zero_grad()
        z_fake = encoder(images)
        x_recon = decoder(z_fake)
        loss_ae = mse_loss(x_recon, images)
        loss_ae.backward()
        optimizer_ae.step()

        # === Train discriminator ===
        optimizer_d.zero_grad()
        z_fake_detached = encoder(images).detach()
        z_real = torch.randn(batch_size, z_dim, device=device) * 5.
        d_real = discriminator(z_real)
        d_fake = discriminator(z_fake_detached)
        labels_real = torch.ones(batch_size, 1, device=device)
        labels_fake = torch.zeros(batch_size, 1, device=device)
        loss_d = bce_loss(d_real, labels_real) + bce_loss(d_fake, labels_fake)
        loss_d.backward()
        optimizer_d.step()

        # === Train generator (encoder) ===
        optimizer_g.zero_grad()
        z_fake_g = encoder(images)
        d_fake_for_g = discriminator(z_fake_g)
        loss_g = bce_loss(d_fake_for_g, labels_real)
        loss_g.backward()
        optimizer_g.step()

        if (i + 1) % 50 == 0:
            print(f"Epoch [{epoch}/{n_epochs}] Batch [{i+1}/{len(dataloader)}] \
                  AE Loss: {loss_ae.item():.4f}, D Loss: {loss_d.item():.4f}, G Loss: {loss_g.item():.4f}")

    # Optionally: save checkpoint or visualize intermediate results

# Visualization: generate grid
encoder.eval(); decoder.eval()
z_values = np.arange(-10, 10, 1.5).astype(np.float32)
nx, ny = len(z_values), len(z_values)
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(nx, ny, wspace=0.05, hspace=0.05)

with torch.no_grad():
    for xi, xv in enumerate(z_values):
        for yi, yv in enumerate(z_values):
            z = torch.tensor([[xv, yv]], device=device)
            x_dec = decoder(z).cpu().numpy().reshape(28, 28)
            ax = fig.add_subplot(gs[xi * ny + yi])
            ax.imshow(x_dec, cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
plt.show()

import torch 
import torch.nn as nn
import torch.optim as optim
import random
import math
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import functional as F

# Problems I encoutnered:
#   - Was dumb as used completed random colors which trained the model to appoximate the mean, aka grey
#   - KDL was killing everything, needed to add a diffusing beta term, problem called the "posterior collapse"

torch.manual_seed(23)

n = 667
noise = 0.08
reds   = torch.cat([torch.clamp(0.85 + noise * torch.randn(n, 1), 0, 1),
                    torch.clamp(       noise * torch.randn(n, 1).abs(), 0, 0.3),
                    torch.clamp(       noise * torch.randn(n, 1).abs(), 0, 0.3)], dim=1)

greens = torch.cat([torch.clamp(       noise * torch.randn(n, 1).abs(), 0, 0.3),
                    torch.clamp(0.85 + noise * torch.randn(n, 1), 0, 1),
                    torch.clamp(       noise * torch.randn(n, 1).abs(), 0, 0.3)], dim=1)

blues  = torch.cat([torch.clamp(       noise * torch.randn(n, 1).abs(), 0, 0.3),
                    torch.clamp(       noise * torch.randn(n, 1).abs(), 0, 0.3),
                    torch.clamp(0.85 + noise * torch.randn(n, 1), 0, 1)], dim=1)

# Stack and convert from [0,1] RGB to [-1,1] for the model
samples = 2 * torch.cat([reds, greens, blues], dim=0) - 1

def sample_to_rgb(sample):
    return ( sample + 1 ) / 2

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.enc  = nn.Linear(3, 2)
        self.enca = nn.ReLU()
        self.mean = nn.Linear(2, 2)
        self.lvar = nn.Linear(2, 2)
        self.dec  = nn.Linear(2, 3)
        self.deca = nn.Tanh()

    def decode(self, z):
        return self.deca( self.dec(z) )

    def forward(self, x):
        r = x
        r = self.enca(self.enc(r))

        mu = self.mean(r)
        lv = self.lvar(r)

        std = torch.exp( 0.5 * lv )
        eps = torch.randn_like(std)
        z = mu + std * eps

        r = self.decode(z)
        return r, mu, lv

class SampleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

training_dataset    = SampleDataset(samples) 
training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)

model       = Model()
optimizer   = optim.Adam( model.parameters(), lr=1e-3 )

def loss_function(preds, target, mu, lvar, beta=0.1):
    mse = F.mse_loss(preds, target)
    #KL Divergence
    kld = -0.5 * torch.mean(1 + lvar - mu.pow(2) - lvar.exp())
    loss = mse + beta * kld
    return loss

for epoch in range(30):
    pbar = tqdm(training_dataloader, desc="Training")
    training_loss = 0
    training_count = 0
    model.train()
    for i, x in enumerate(pbar):
        optimizer.zero_grad()

        preds, mu, lvar = model(x)
        loss = loss_function(preds, x, mu, lvar)
        loss.backward()
        optimizer.step()

        training_count += 1
        training_loss += loss.item()
        loss_str = f"{(training_loss/training_count):.4f}"
        pbar.set_postfix({"Loss":loss_str})

model.eval()
with torch.no_grad():
    # Encode all samples to get their latent positions
    all_preds, all_mu, _ = model(samples)
    all_colors = sample_to_rgb(samples).clamp(0, 1).numpy()
    mu_np = all_mu.numpy()

    # Sweep latent space on a grid to see decoded colors
    grid_size = 20
    z1 = torch.linspace(-3, 3, grid_size)
    z2 = torch.linspace(-3, 3, grid_size)
    grid_z = torch.stack(torch.meshgrid(z1, z2, indexing='ij'), dim=-1).reshape(-1, 2)
    grid_colors = sample_to_rgb(model.decode(grid_z)).clamp(0, 1).numpy()
    grid_img = grid_colors.reshape(grid_size, grid_size, 3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: latent space scatter, each point colored by its RGB value
ax1.scatter(mu_np[:, 0], mu_np[:, 1], c=all_colors, s=8, alpha=0.6)
ax1.set_title("Latent Space (colored by RGB)")
ax1.set_xlabel("z[0]")
ax1.set_ylabel("z[1]")

# Right: grid of decoded colors across latent space
ax2.imshow(grid_img, origin='lower', extent=[-3, 3, -3, 3], aspect='auto')
ax2.set_title("Decoded Colors Across Latent Space")
ax2.set_xlabel("z[0]")
ax2.set_ylabel("z[1]")

plt.tight_layout()
plt.show()
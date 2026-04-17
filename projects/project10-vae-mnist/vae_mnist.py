import torch 
import torch.nn as nn
import torch.optim as optim
import random
import math
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import functional as F

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ])

batch_size = 32

mnist_train = torchvision.datasets.MNIST(root='./data',      train=True, download=True, transform=transform)
mnist_validation  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
training_loader     = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
validation_loader   = DataLoader(mnist_validation, batch_size=batch_size, shuffle=True)

LATENT_DIM = 32

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Size = (B, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Size = (B, 32, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Size = (B, 64, 7, 7) -> bottleneck
        flattened_size = 64 * 7 * 7
        self.mean = nn.Linear(flattened_size, LATENT_DIM)
        self.lvar = nn.Linear(flattened_size, LATENT_DIM)

        # Project back up from latent to spatial
        self.decode_fc = nn.Linear(LATENT_DIM, flattened_size)

        self.up1     = nn.Upsample( scale_factor=2, mode='bilinear', align_corners=False )
        self.dcn1    = nn.Conv2d( in_channels=64, out_channels=32, kernel_size=3, padding=1 )
        self.dcn1a   = nn.ReLU()

        self.up2     = nn.Upsample( scale_factor=2, mode='bilinear', align_corners=False )
        self.dcn2    = nn.Conv2d( in_channels=32, out_channels=1, kernel_size=3, padding=1 )

    def forward(self, x):
        B = x.shape[0]

        r = x
        r = self.pool1(F.relu(self.conv1(r)))
        r = self.pool2(F.relu(self.conv2(r)))

        r = torch.flatten(r, 1)
        mu = self.mean(r)
        lv = self.lvar(r)

        std = torch.exp( 0.5 * lv )
        eps = torch.randn_like(std)
        z = mu + std * eps          # reparameterization trick

        r = F.relu(self.decode_fc(z))
        r = r.reshape(B, 64, 7, 7)

        r = self.up1(r)
        r = self.dcn1a(self.dcn1(r))

        r = self.up2(r)
        r = self.dcn2(r)
        return r, mu, lv

def loss_function(preds, targets, mu, lvar, beta=0.15):
    # Reconstruction loss
    mse = F.mse_loss(preds, targets)
    # KL divergence
    kld = -0.5 * torch.mean(1 + lvar - mu.pow(2) - lvar.exp())
    loss = mse + beta * kld
    return loss

model       = Model()
optimizer   = optim.Adam( model.parameters(), lr=1e-3 )

for epoch in range(1):

    model.train()
    pbar = tqdm(training_loader, "Training")
    training_loss = 0
    training_count = 0
    for i, (x, _) in enumerate(pbar):
        optimizer.zero_grad()

        preds, mu, lvar = model(x)
        loss = loss_function(preds, x, mu, lvar)
        loss.backward()
        optimizer.step()

        training_loss+= loss
        training_count+=1
        loss_str = f"{(training_loss / training_count):.5f}"
        pbar.set_postfix({"loss":loss_str})

    model.eval()
    validation_loss = 0
    validation_count = 0
    pbar = tqdm(validation_loader, desc="Validatn")
    with torch.no_grad():
        for i, (x, y) in  enumerate(pbar):
            preds, mu, lvar = model(x)
            loss = loss_function(preds, x, mu, lvar)
            validation_loss+= loss
            validation_count+=1
            loss_str = f"{(validation_loss / validation_count):.5f}"
            pbar.set_postfix({"loss":loss_str})

def visualize_reconstructions(model, loader, n=8):
    """Show n original images alongside their VAE reconstructions."""
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n]
    with torch.no_grad():
        recon, _, _ = model(x)

    # Denormalize from [-1, 1] back to [0, 1]
    x     = (x     * 0.5 + 0.5).clamp(0, 1)
    recon = (recon * 0.5 + 0.5).clamp(0, 1)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i, 0].cpu(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_title('Original', loc='left')
    axes[1, 0].set_title('Reconstructed', loc='left')
    plt.tight_layout()
    plt.savefig('./projects/project10-vae-mnist/reconstructions.png')
    #plt.show()

def visualize_samples(model, n=8):
    """Decode random latent vectors sampled from N(0, I)."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, LATENT_DIM)
        B = z.shape[0]
        r = F.relu(model.decode_fc(z))
        r = r.reshape(B, 64, 7, 7)
        r = model.up1(r)
        r = model.dcn1a(model.dcn1(r))
        r = model.up2(r)
        samples = model.dcn2(r)

    samples = (samples * 0.5 + 0.5).clamp(0, 1)

    fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 2))
    for i in range(n):
        axes[i].imshow(samples[i, 0].cpu(), cmap='gray')
        axes[i].axis('off')
    fig.suptitle('Random Samples from Latent Space')
    plt.tight_layout()
    plt.savefig('./projects/project10-vae-mnist/samples.png')
    #plt.show()

visualize_reconstructions(model, validation_loader)
visualize_samples(model)
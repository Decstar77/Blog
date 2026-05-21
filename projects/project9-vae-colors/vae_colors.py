import torch
import torch.nn as nn
from torch.nn import functional as F

# Problems I encountered:
#   - Was dumb as used completely random colors which trained the model to
#     approximate the mean, aka grey.
#   - KL divergence was killing everything; needed a diffusing beta term —
#     the problem is called "posterior collapse".

torch.manual_seed(23)

# ── Dataset ───────────────────────────────────────────────────────
# Three blobs of near-pure red / green / blue, jittered with a little
# noise. Cheap and deterministic, so it lives at module level.

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
    """Map the model's [-1,1] output back to [0,1] RGB."""
    return (sample + 1) / 2


# ── Model ─────────────────────────────────────────────────────────

class Model(nn.Module):
    """A tiny VAE: 3-D RGB <-> 2-D latent space.

    The runner looks for a class named `Model` in each project file.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.enc  = nn.Linear(3, 2)
        self.enca = nn.ReLU()
        self.mean = nn.Linear(2, 2)
        self.lvar = nn.Linear(2, 2)
        self.dec  = nn.Linear(2, 3)
        self.deca = nn.Tanh()

    def encode(self, x):
        r = self.enca(self.enc(x))
        return self.mean(r), self.lvar(r)

    def decode(self, z):
        return self.deca(self.dec(z))

    def forward(self, x):
        mu, lv = self.encode(x)

        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        z = mu + std * eps

        return self.decode(z), mu, lv


def loss_function(preds, target, mu, lvar, beta=0.1):
    mse = F.mse_loss(preds, target)
    # KL divergence between the encoder posterior and a unit Gaussian.
    kld = -0.5 * torch.mean(1 + lvar - mu.pow(2) - lvar.exp())
    return mse + beta * kld


# ── Inference helpers (used by the runner) ────────────────────────

@torch.no_grad()
def latent_scatter(net):
    """Encode every training colour to its latent mean.

    Returns a list of {z0, z1, r, g, b} dicts.
    """
    net.eval()
    mu, _ = net.encode(samples)
    colors = sample_to_rgb(samples).clamp(0, 1)
    return [{"z0": float(z[0]), "z1": float(z[1]),
             "r": float(c[0]), "g": float(c[1]), "b": float(c[2])}
            for z, c in zip(mu, colors)]


@torch.no_grad()
def decode_grid(net, grid_size=24, span=3.0):
    """Sweep a grid_size x grid_size grid of the latent space and decode
    each point to an RGB colour. Returns a nested [row][col][rgb] list.
    """
    net.eval()
    axis = torch.linspace(-span, span, grid_size)
    grid_z = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), dim=-1).reshape(-1, 2)
    colors = sample_to_rgb(net.decode(grid_z)).clamp(0, 1)
    colors = colors.reshape(grid_size, grid_size, 3)
    return [[[round(float(v), 4) for v in px] for px in row] for row in colors]


@torch.no_grad()
def decode_point(net, z0, z1):
    """Decode a single latent coordinate to an RGB colour in [0,1]."""
    net.eval()
    z = torch.tensor([[z0, z1]], dtype=torch.float32)
    rgb = sample_to_rgb(net.decode(z)).clamp(0, 1)[0]
    return [round(float(v), 4) for v in rgb]


# ── Training ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Dataset

    class SampleDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, index):
            return self.data[index]

    training_dataloader = DataLoader(SampleDataset(samples), batch_size=32, shuffle=True)

    model     = Model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(30):
        pbar = tqdm(training_dataloader, desc=f"Epoch {epoch}")
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
            pbar.set_postfix({"Loss": f"{(training_loss / training_count):.4f}"})

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "model.pt"))

    # ── Visualise ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        all_mu, _ = model.encode(samples)
        all_colors = sample_to_rgb(samples).clamp(0, 1).numpy()
        mu_np = all_mu.numpy()

        grid_size = 20
        axis = torch.linspace(-3, 3, grid_size)
        grid_z = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), dim=-1).reshape(-1, 2)
        grid_colors = sample_to_rgb(model.decode(grid_z)).clamp(0, 1).numpy()
        grid_img = grid_colors.reshape(grid_size, grid_size, 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(mu_np[:, 0], mu_np[:, 1], c=all_colors, s=8, alpha=0.6)
    ax1.set_title("Latent Space (colored by RGB)")
    ax1.set_xlabel("z[0]")
    ax1.set_ylabel("z[1]")

    ax2.imshow(grid_img, origin='lower', extent=[-3, 3, -3, 3], aspect='auto')
    ax2.set_title("Decoded Colors Across Latent Space")
    ax2.set_xlabel("z[0]")
    ax2.set_ylabel("z[1]")

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "latent_space.png"), dpi=120)
    plt.show()

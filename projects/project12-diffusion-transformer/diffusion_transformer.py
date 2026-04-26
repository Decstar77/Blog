import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import math
import random

import torchvision
import numpy as np
from tqdm import tqdm

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ])

mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_validation  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
batch_size = 64

beta_start = 1e-4
beta_end = 0.02
tmax = 1000
torch.manual_seed(23)
random.seed(32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

def f_beta(i): 
    return beta_start + ( beta_end - beta_start ) * ( i / ( tmax - 1 ) )

def f_alpha(i):
    return 1 - f_beta(i)

def sinusoidal( t, d_model ):
        N = 10_000
        D = d_model
        K = torch.arange(0, D//2)

        positions = t.unsqueeze(1).float()
        embeddings = torch.zeros(t.shape[0], D, device=t.device)
        K = K.to(t.device)
        denominators = torch.pow( N, 2 * K / D )
        embeddings[:, 0::2] = torch.sin(positions/denominators)
        embeddings[:, 1::2] = torch.cos(positions/denominators)
        return embeddings

def resize_modulation_vecs(vecs):
    return tuple(v.unsqueeze(1) for v in vecs)

class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio = 4):
        super(DiTBlock, self).__init__()
        self.adal       = nn.Linear(hidden, 6 * hidden)
        self.adaa       = nn.SiLU()
        self.preln      = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn       = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.mlpln     = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp        = nn.Sequential(
            nn.Linear(hidden, hidden*mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden*mlp_ratio, hidden)
        )

    def forward(self, x):
        # x -> [B, T, H]

        x, c = x
        shift1, scale1, gate1, shift2, scale2, gate2 = resize_modulation_vecs( self.adal( self.adaa( c ) ).chunk(6, dim=-1) )

        # Phase 1
        h = self.preln(x) * ( 1 + scale1 ) + shift1
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1 * a

        # Phase 2
        h = self.mlpln(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)

        return x

class DitFinalModule(nn.Module):
    def __init__(self, hidden, patch_size):
        super(DitFinalModule, self).__init__()
        self.H          = hidden
        self.P          = patch_size
        self.adal       = nn.Linear(hidden, 2 * hidden)
        self.adaa       = nn.SiLU()
        self.norm       = nn.LayerNorm(hidden, elementwise_affine=False)
        self.proj       = nn.Linear( hidden, 1 * patch_size * patch_size ) 

    def forward(self, x):
        x, c = x
        shift, scale = resize_modulation_vecs( self.adal( self.adaa( c ) ).chunk(2, dim=-1) )
        r = self.norm(x) * ( 1 + scale ) + shift
        r = self.proj(r)

        # [B, T, P * P * 1] - > # [B, C, 28, 28]
        P = int(self.P)
        B = x.shape[0]
        T = x.shape[1]
        H = W = int(math.sqrt(T))
        r = r.reshape(B, H, W, P, P, 1)
        r = r.permute(0, 5, 1, 3, 2, 4)
        r = r.reshape(B, 1, H*P, W*P)
        return r

class Model(nn.Module):
    def __init__(self, patch_size=4, hidden=192, depth=6, heads=6, classes=10):
        super(Model, self).__init__()
        self.patch_size = patch_size
        self.hidden = hidden
        self.num_patches = int(28 / patch_size) ** 2
        
        self.patch       = nn.Conv2d(1, hidden, kernel_size=patch_size, stride=patch_size)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches, hidden))
        self.time_mlp    = nn.Sequential( nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden) )
        self.label_embed = nn.Embedding(classes, hidden)
        self.blocks      = nn.ModuleList([DiTBlock(hidden, heads) for _ in range(depth)])
        self.final       = DitFinalModule(hidden, patch_size)

    def forward(self, x):
        x, t, l = x

        # [B, C, 28, 28] -> [B, Hidden, 7, 7] -> [B, H, 49] -> [B, 49, H]
        r = self.patch(x).flatten(-2).transpose(1, 2) + self.pos_embed
        c = self.time_mlp( sinusoidal(t, self.hidden) ) + self.label_embed(l)

        for blk in self.blocks: r = blk((r,c))
        r = self.final((r, c))

        return r

training_loader     = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
validation_loader   = DataLoader(mnist_validation, batch_size=batch_size, shuffle=True)
beta_tensor         = torch.tensor( [ f_beta( i ) for i in range( tmax ) ], dtype=torch.float32, device=device )
alpha_tensor        = torch.tensor( [ f_alpha( i ) for i in range( tmax ) ], dtype=torch.float32, device=device )
alpha_prod_tensor   = torch.cumprod( alpha_tensor, dim=0 )

epochs          = 20
model           = Model().to(device)
loss_function   = nn.MSELoss()
optimizer       = optim.Adam(model.parameters(), lr=3e-4)
scheduler       = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

for blk in model.blocks:
    nn.init.zeros_(blk.adal.weight)

nn.init.zeros_(model.final.adal.weight)
nn.init.zeros_(model.final.proj.weight)
nn.init.zeros_(model.final.proj.bias)

state_dict = torch.load("./projects/project12-diffusion-transformer/model.pt", weights_only=True, map_location=device)
model.load_state_dict(state_dict)

total_params = sum(p.numel() for p in model.parameters())
print(f"DiT! {(total_params / 1000000):.2f} million parameters ")

for epoch in range(0):
    print(f"==================={epoch}===================")
    model.train()
    training_loss = 0
    training_count = 0
    pbar = tqdm(training_loader, desc="Training")
    for i, (x0, y) in  enumerate(pbar):
        x0 = x0.to(device)
        y = y.to(device)
        B = x0.shape[0]

        tvals = torch.randint(0, tmax, (B,), device=device)
        alphas = alpha_prod_tensor[tvals].float().view(B, 1, 1, 1)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(alphas) * x0 + torch.sqrt(1 - alphas) * eps

        preds = model((xt, tvals, y))
        optimizer.zero_grad()
        loss = loss_function(preds, eps)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        training_loss+= loss
        training_count+=1
        loss_str = f"{(training_loss / training_count):.5f}"
        pbar.set_postfix({"loss":loss_str})

    scheduler.step()

    model.eval()
    validation_loss = 0
    validation_count = 0
    pbar = tqdm(validation_loader, desc="Validatn")
    with torch.no_grad():
        for i, (x, y) in  enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            B = x.shape[0]
            e_noise = torch.randn_like(x) # [64, 1, 28, 28]
            t_en = torch.randint(low=0, high=tmax, size=(B,), dtype=torch.long, device=device)
            alpha_bar_t = alpha_prod_tensor[t_en].float().view(B, 1, 1, 1)  # [64, 1, 1, 1]
            x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * e_noise
            preds = model((x_t, t_en, y))
            loss = loss_function(preds, e_noise)
            validation_loss+= loss
            validation_count+=1
            loss_str = f"{(validation_loss / validation_count):.5f}"
            pbar.set_postfix({"loss":loss_str})

    torch.save(model.state_dict(), "./projects/project12-diffusion-transformer/model.pt")

@torch.no_grad()
def sample_reverse(num_samples, value):
    model.eval()
    x_t = torch.randn(num_samples, 1, 28, 28, device=device)

    for t in reversed(range(tmax)):
        t_en = torch.full((num_samples,), t, dtype=torch.long, device=device)
        label = torch.full((num_samples,), value, dtype=torch.long, device=device)
        eps_hat = model( (x_t, t_en, label) )

        alpha_t = alpha_tensor[t]
        alpha_bar_t = alpha_prod_tensor[t]
        beta_t = beta_tensor[t]

        # DDPM reverse mean: mu_theta(x_t, t)
        term1 = (1.0 / torch.sqrt(alpha_t))
        term2 =( 1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
        mu_t = term1 * ( x_t - term2 * eps_hat )

        if t > 0:
            z = torch.randn_like(x_t)
            x_t = mu_t + torch.sqrt(beta_t) * z
        else:
            x_t = mu_t

    return x_t

num_digits = 10
num_per_digit = 10
fig, axes = plt.subplots(num_per_digit, num_digits, figsize=(num_digits * 2, num_per_digit * 2))

for d in tqdm(range(num_digits), desc="Generating samples"):
    samples = sample_reverse(num_per_digit, d)
    for r in range(num_per_digit):
        image = samples[r].reshape(28, 28).cpu()
        image = (image + torch.ones(28, 28)) / 2
        axes[r][d].imshow(image, cmap='gray')
        axes[r][d].axis('off')

plt.tight_layout()
print("saving image")
plt.savefig('./projects/project12-diffusion-transformer/sample.png')




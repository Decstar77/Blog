import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optimorch
from torch.utils.data import DataLoader, Dataset

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

def f_beta(i): 
    return beta_start + ( beta_end - beta_start ) * ( i / ( tmax - 1 ) )

def f_alpha(i):
    return 1 - f_beta(i)


class Model(nn.Module):
    def __init__(self, patch_size):
        super(Model, self).__init__()
        
        self.channels = [32, 64, 128]
        
        self.time_embed  = nn.Linear(32, 32)
        self.label_embed = nn.Embedding(10, 32)
        
        self.enc1c = nn.Conv2d(1, self.channels[0], kernel_size=3, padding=1)
        self.enc1p = nn.Linear( 32, self.channels[0] )
        self.enc1n = nn.GroupNorm( 8, self.channels[0] )
        self.enc1a = nn.ReLU()
        self.enc1m = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2c = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, padding=1)
        self.enc2p = nn.Linear( 32, self.channels[1] )
        self.enc2n = nn.GroupNorm( 8, self.channels[1] )
        self.enc2a = nn.ReLU()
        self.enc2m = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3c = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, padding=1)
        self.enc3p = nn.Linear( 32, self.channels[2] )
        self.enc3n = nn.GroupNorm( 8, self.channels[2] )
        self.enc3a = nn.ReLU()
        self.enc3m = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.botcv  = nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[2], kernel_size=3, padding=1)
        self.botav  = nn.ReLU()
        
        self.up3     = nn.Upsample( scale_factor=2, mode='bilinear', align_corners=False )
        self.dcn3    = nn.Conv2d( in_channels=self.channels[2] * 2, out_channels=self.channels[1], kernel_size=3, padding=1 )
        self.dcn3n   = nn.GroupNorm(8, self.channels[1])
        self.dcn3p   = nn.Linear(32, self.channels[1] )
        self.dcn3a   = nn.ReLU()

        self.up2     = nn.Upsample( scale_factor=2, mode='bilinear', align_corners=False )
        self.dcn2    = nn.Conv2d( in_channels=self.channels[1] * 2, out_channels=self.channels[0], kernel_size=3, padding=1 )
        self.dcn2n   = nn.GroupNorm(8, self.channels[0])
        self.dcn2p   = nn.Linear(32, self.channels[0] )
        self.dcn2a   = nn.ReLU()

        self.up1     = nn.Upsample( scale_factor=2, mode='bilinear', align_corners=False )
        self.dcn1    = nn.Conv2d( in_channels=self.channels[0] * 2, out_channels=1, kernel_size=3, padding=1 )

    def sinusoidal( self, t, d_model ):
        N = 10_000
        D = d_model
        K = torch.arange(0, D//2)

        positions = t.unsqueeze(1).float()
        embeddings = torch.zeros(t.shape[0], D)
        denominators = torch.pow( N, 2 * K / D ) 
        embeddings[:, 0::2] = torch.sin(positions/denominators)
        embeddings[:, 1::2] = torch.cos(positions/denominators)
        return embeddings

    def forward(self, x):
        r, t, l = x

        t_embed = self.time_embed( self.sinusoidal(t, 32) )
        l_embed = self.label_embed( l )
        embed = t_embed + l_embed

        r = self.enc1c(r)
        r = r + self.enc1p( embed ).unsqueeze(-1).unsqueeze(-1)
        r = self.enc1n(r)
        s1 = self.enc1a(r)
        r = self.enc1m(s1)
        
        r = self.enc2c(r)
        r = r + self.enc2p( embed ).unsqueeze(-1).unsqueeze(-1)
        r = self.enc2n(r)
        s2 = self.enc2a(r)
        r = self.enc2m(s2)
        
        r = self.enc3c(r)
        r = r + self.enc3p( embed ).unsqueeze(-1).unsqueeze(-1)
        r = self.enc3n(r)
        s3 = self.enc3a(r)
        r = self.enc3m(s3)

        r = self.botav( self.botcv(r) )

        r = self.up3(r)
        if r.shape[-2:] != s3.shape[-2:]:
            r = torch.nn.functional.interpolate(r, size=s3.shape[-2:], mode='bilinear', align_corners=False)

        r = torch.cat([r, s3], dim=1)
        r = self.dcn3( r )
        r = self.dcn3n( r )
        r =  r + self.dcn3p( embed ).unsqueeze(-1).unsqueeze(-1)
        r = self.dcn3a( r )

        r = self.up2(r)
        r = torch.cat([r, s2], dim=1)
        r = self.dcn2( r )
        r = self.dcn2n( r )
        r =  r + self.dcn2p( embed ).unsqueeze(-1).unsqueeze(-1)
        r = self.dcn2a( r )

        r = self.up1(r)
        r = torch.cat([r, s1], dim=1)
        r = self.dcn1( r )

        return r

training_loader     = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
validation_loader   = DataLoader(mnist_validation, batch_size=batch_size, shuffle=True)
beta_tensor         = torch.tensor( [ f_beta( i ) for i in range( tmax ) ], dtype=torch.float32 )
alpha_tensor        = torch.tensor( [ f_alpha( i ) for i in range( tmax ) ], dtype=torch.float32 )
alpha_prod_tensor   = torch.cumprod( alpha_tensor, dim=0 )

model           = Model()
loss_function   = nn.MSELoss()
optimizer       = optim.Adam(model.parameters(), lr=1e-4)
scheduler       = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

for epoch in range(0):
    print(f"==================={epoch}===================")
    model.train()
    training_loss = 0
    training_count = 0
    pbar = tqdm(training_loader, desc="Training")
    for i, (x0, y) in  enumerate(pbar):
        B = x0.shape[0]

        tvals = torch.randint(0, tmax, (B,))
        alphas = alpha_prod_tensor[tvals].float().view(B, 1, 1, 1)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(alphas) * x0 + torch.sqrt(1 - alphas) * eps

        preds = model((xt, tvals, y))

        optimizer.zero_grad()
        loss = loss_function(preds, eps)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            B = x.shape[0]
            e_noise = torch.randn_like(x) # [64, 1, 28, 28]
            t_en = torch.randint(low=0, high=tmax, size=(B,), dtype=torch.long)
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
    x_t = torch.randn(num_samples, 1, 28, 28)

    for t in reversed(range(tmax)):
        t_en = torch.full((num_samples,), t, dtype=torch.long)
        label = torch.full((num_samples,), value, dtype=torch.long) 
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






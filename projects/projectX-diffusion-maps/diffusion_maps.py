import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optimorch
from torch.utils.data import DataLoader, Dataset
import math
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

beta_start = 1e-4
beta_end = 0.02
t_max = 1000

torch.manual_seed(23)
random.seed(32)

def f_beta(i):
    return beta_start + ( beta_end - beta_start ) * ( i / ( t_max - 1 ) )

def f_alpha( i ):
    return 1 - f_beta(i)

def encode_map(grid):
    num_classes = 5
    one_hot = (np.arange(num_classes)[:, None, None] == grid[None, :, :]).astype(np.float32)
    norm = (one_hot * 2) - 1
    return torch.from_numpy(norm)

def decode_map(tensor):
    # tensor: (num_classes, W, H) in [-1, 1] — argmax over class dim gives label grid
    return tensor.argmax(dim=0).numpy()

COLORS = [
    '#1a1a2e',  # 0 empty air    - near-black navy
    '#4a4a5a',  # 1 solid wall   - dark slate
    '#c8a96e',  # 2 floor        - warm sand
    '#00e676',  # 3 player spawn - bright green
    '#ff1744',  # 4 enemy spawn  - vivid red
]

LABELS = ['Empty', 'Wall', 'Floor', 'Player', 'Enemy']
def display_map(grid):
    cmap = ListedColormap(COLORS)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(COLORS) - 1,
              interpolation='nearest', aspect='equal')

    legend = [mpatches.Patch(color=COLORS[i], label=f'{i} – {LABELS[i]}')
              for i in range(len(COLORS))]
    ax.legend(handles=legend, loc='upper right', fontsize=9,
              framealpha=0.85, edgecolor='#888')

    ax.set_title("Map", fontsize=11)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    plt.show()

class GridDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return encode_map( self.data[index] )

class EncoderBlock(nn.Module):
    def __init__(self, embed_vec, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linn = nn.Linear(embed_vec, out_channels)
        self.lina = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        r, embed = x
        r = self.conv(r)
        self.residual = self.lina( r + self.linn( embed ).unsqueeze(-1).unsqueeze(-1))
        r = self.pool(self.residual)
        return r

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu = True):
        super(DecoderBlock, self).__init__()
        self.relu = relu
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        if ( relu ):
            self.lina = nn.ReLU()

    def forward(self, x):
        r, residual = x 
        r = self.up(r)
        r = torch.cat( [r, residual], dim=1 )
        r = self.conv(r)
        if self.relu:
            r = self.lina(r)
        return r

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(t_max, 32)
        self.e1 = EncoderBlock(embed_vec=32, in_channels=5,  out_channels=32)
        self.e2 = EncoderBlock(embed_vec=32, in_channels=32, out_channels=64)
        self.d1 = DecoderBlock(in_channels=64 * 2, out_channels=32)
        self.d2 = DecoderBlock(in_channels=32 * 2, out_channels=5, relu=False)

    def forward(self, x):
        r, t = x
        embed = self.embed( t )

        r = self.e1( ( r, embed ) )
        r = self.e2( ( r, embed ) )
        r = self.d1( ( r, self.e2.residual ) )
        r = self.d2( ( r, self.e1.residual ) )

        return r

if __name__ == "__main__":
    from mapgen import Cellular, Drunk, BSP, generate_map_data
    import os
    import numpy as np

    maps_path = os.path.join( os.path.dirname(__file__), "maps.npz" )
    if os.path.exists(maps_path) == False:
        cellular = Cellular()
        drunk = Drunk()
        bsp = BSP()
        map_grids = generate_map_data(maps_path, cellular, drunk, bsp)
    else:
        print("Loading data...")
        data = np.load(maps_path)
        #map_grids = [data[key] for key in data.files[0:2000]]
        map_grids = [data[key] for key in data.files]

    training_dataset        = GridDataset(map_grids[:int( 0.9 * len(map_grids) )])
    training_dataloader     = DataLoader(dataset=training_dataset,     batch_size=32, shuffle=True)
    validation_dataset      = GridDataset(map_grids[int(0.9 * len(map_grids)):])
    validation_dataloader   = DataLoader(dataset=validation_dataset,   batch_size=32)

    model       = Model()
    loss_func   = nn.MSELoss()
    optimizer   = optim.Adam( model.parameters(), lr=0.001 )

    beta_tensor         = torch.tensor([ f_beta( i ) for i in range(t_max) ], dtype=float )
    alpha_tensor        = torch.tensor([ f_alpha( i ) for i in range(t_max) ], dtype=float )
    alpha_bar_tensor    = torch.cumprod( alpha_tensor, dim=0 )

    def compute_xt_t(x0, B):
        epsilon = torch.randn_like( x0 )
        t = torch.randint(low=0, high=t_max, size=(B,), dtype=torch.long)
        alpha_bar_t = alpha_bar_tensor[t].reshape(B, 1, 1, 1)
        xt = torch.sqrt( alpha_bar_t ) * x0  + torch.sqrt( 1 - alpha_bar_t ) * epsilon
        return xt.float(), t, epsilon
    
    @torch.no_grad()
    def sample_reverse(num_samples):
        model.eval()
        x_t = torch.randn(num_samples, 5, 32, 32)

        for t in reversed(range(t_max)):
            t_en = torch.full((num_samples,), t, dtype=torch.long)
            eps_hat = model( (x_t, t_en) )

            alpha_t = alpha_tensor[t]
            alpha_bar_t = alpha_bar_tensor[t]
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

    grid = decode_map(sample_reverse(1))
    print(grid.shape)
    display_map(grid)

    for epoch in range(1):
        print(f"============Epoch={epoch}============")

        training_loss = 0
        training_count = 0
        model.train()
        pbar = tqdm(training_dataloader, desc="Training")
        for i, x0 in enumerate(pbar):
            optimizer.zero_grad()
            xt, t, epsilon = compute_xt_t(x0, x0.shape[0])
            preds = model((xt, t))
            loss = loss_func(preds, epsilon)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_count += 1
            loss_str = f"{(training_loss / training_count):.4f}"
            pbar.set_postfix({"Loss" : loss_str})

        validation_loss = 0
        validation_count = 0
        model.eval()
        pbar = tqdm(validation_dataloader, desc="Validatn")
        for i, x0 in enumerate(pbar):
            xt, t, epsilon = compute_xt_t(x0, x0.shape[0])
            preds = model((xt, t))
            loss = loss_func(preds, epsilon)
            validation_loss += loss.item()
            validation_count += 1
            loss_str = f"{(validation_loss / validation_count):.4f}"
            pbar.set_postfix({"Loss" : loss_str})

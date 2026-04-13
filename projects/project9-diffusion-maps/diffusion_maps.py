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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x):
        pass

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
        map_grids = [data[key] for key in data.files[0:10]]

    dataset     = GridDataset(map_grids)
    dataloader  = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    model       = Model()
    loss_func   = nn.MSELoss()
    #optimizer   = optim.Adam( model.parameters(), lr=0.001 )

    alpha_tensor        = torch.tensor([ f_alpha( i ) for i in range(t_max) ], dtype=float )
    alpha_bar_tensor    = torch.cumprod( alpha_tensor, dim=0 )
    
    for epoch in range(1):
        for i, x in enumerate(dataloader):
            B = x.shape[0]
            #print(x.shape)
            p = decode_map(x[0])
            display_map(p)
            #print(x[0][1])
            #epsilon = torch.randn_like( x )
            #t_value = torch.randint(low=0, high=t_max, size=(B,), dtype=torch.long)
            #print(epsilon.shape)
            break


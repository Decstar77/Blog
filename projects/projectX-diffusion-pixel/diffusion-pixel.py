import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optimorch
from torch.utils.data import DataLoader, Dataset
import math
import random
from tqdm import tqdm
import numpy as np
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ebrahimelgazar/pixel-art", output_dir="data/pixel-art")

hex_pallete = []
with open("projects/projectX-diffusion-pixel/hept32.txt", 'r') as file:
    def convert_color(hex_color):
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
    
    for line in file:
        hex_pallete.append( convert_color( line.strip() ) )

np_sprites = np.load("data/pixel-art/sprites.npy")
np_lables = np.load("data/pixel-art/sprites_labels.npy")

class SpriteDataset(Dataset):
    def __init__(self, sprites):
        self.sprites = sprites
        
    def __len__(self):
        return len(self.sprites)
    
    def __getitem__(self, index):
        pass

value = torch.tensor( np_sprites[0] )


for x in range(value.shape[0]):
    for y in range(value.shape[1]):
        c  = value[x][y]
        print(c)
        break
    break


num_tokens = len( hex_pallete ) + 1
MASK_ID = len( hex_pallete )

class MaskGiT(nn.Module):
    def __init__(self, d_model = 256, heads=4, depth=4):
        super(MaskGiT, self).__init__()
        #self.pos_embed = nn.Parameter( torch.zeros() )
        self.token_embed    = nn.Embedding(num_tokens, d_model)
        self.blocks         = nn.ModuleList( [ nn.TransformerEncoderLayer(d_model, heads, batch_first=True) for _ in range(depth) ] )
        self.proj           = nn.Linear(d_model, num_tokens - 1 )
        
    def forward(self, x):
        x, t = x
        r = self.token_embed(x)
        for blk in self.blocks: r, _ = blk(r,r,r)
        r = self.proj(r)
        return r

for epoch in range(1):
    pass






import torch 
import torch.nn as nn
import torch.optim as optim
import random
import math
from torch.utils.data import DataLoader
import torch.optim as optimorch 
from tqdm import tqdm

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
        data = np.load(maps_path)
        map_grids = [data[key] for key in data.files]

    print(len(map_grids))
    
    model = Model()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam()

    


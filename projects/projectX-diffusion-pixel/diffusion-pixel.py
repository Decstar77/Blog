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

print("Path to dataset files:", path)


#t1 = torch.tensor([[2, 2],[3, 3]])










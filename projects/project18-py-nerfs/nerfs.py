import torch, torch.nn as nn, numpy as np, os
import torchvision
import itertools
from tqdm import tqdm

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

print("Using device:", device)

import kagglehub
path = kagglehub.dataset_download("huabrother/tiny-nerf-data" )

print("Path to dataset files:", path)

data = np.load(os.path.join(path, "tiny_nerf_data.npz"))
images, poses, focal = data["images"], data["poses"], float(data["focal"])
img_height, img_width = images.shape[1:3]

images = torch.tensor(images, device=device)
poses  = torch.tensor(poses, device=device, dtype=torch.float32)
test_img, test_pose = images[101], poses[101]
images, poses = images[:100], poses[:100]  # hold last one out for eval

L = 10
INPUT_DIM = 2 + 4 * L

def positional_encode(x):
    f = 2 ** torch.arange(L)
    i = x.unsqueeze(-1) * f * torch.pi
    s = torch.cat([torch.sin(i), torch.cos(i)], dim=-1)
    return torch.cat([x, torch.flatten(s, start_dim=-2)], dim=-1)

class ImageRegressionTest(nn.Module):
    def __init__(self):
        super(ImageRegressionTest, self).__init__()
        self.d = 256
        self.layer1 = nn.Linear(INPUT_DIM, self.d)
        self.layer2 = nn.Linear(self.d, self.d)
        self.layer3 = nn.Linear(self.d, 3)
        
    def forward(self, x):
        x = torch.relu( self.layer1( x ) )
        x = torch.relu( self.layer2( x ) )
        return self.layer3(x)

def train_regression_model():
    model = ImageRegressionTest()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    vs = ( torch.arange(img_height) + 0.5 ) / img_height
    us = ( torch.arange(img_width) + 0.5 ) / img_width
    v_grid, u_grid = torch.meshgrid(vs, us, indexing='ij')
    uvs = torch.stack([v_grid, u_grid], dim=-1)
    uvs = positional_encode(uvs)

    pbar = tqdm(range(1000), desc="Training")
    for epoch in enumerate(pbar):
        optimizer.zero_grad()
        cols = model(uvs)
        loss = criterion(cols, test_img)
        loss.backward()
        optimizer.step()

        loss_str = f"{(loss.item()):.4f}"
        pbar.set_postfix({"Loss":loss_str})

    with torch.no_grad():
        model.eval()
        cols = model(uvs)
        cols = cols.permute(2, 0, 1)
        img = test_img.permute(2, 0, 1)
        torchvision.utils.save_image(cols, "projects/project18-py-nerfs/sample.png")
        torchvision.utils.save_image(img,  "projects/project18-py-nerfs/sample_r.png")




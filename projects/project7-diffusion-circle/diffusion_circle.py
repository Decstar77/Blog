import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np

from sklearn.datasets import make_circles

samples = 2000
noisy_circles_x, noisy_circles_y = make_circles( n_samples=samples, noise=0.05, random_state=6)
scaled_circles_x = (noisy_circles_x - np.mean(noisy_circles_x, axis=0)) / np.std(noisy_circles_x,axis=0)

def visualize(X ,y):
    plt.figure(figsize=(7, 7))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='navy', label='Outer Circle (y=0)', alpha=0.6)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='darkorange', label='Inner Circle (y=1)', alpha=0.6)
    plt.title("Visualizing sklearn.datasets.make_circles")
    plt.xlabel("Feature 1 (X coordinate)")
    plt.ylabel("Feature 2 (Y coordinate)")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.show()

beta_start = 1e-4
beta_end = 0.02
t_max_steps = 100
random.seed( 46 )

def f_beta(i): # basic lerp function
    return beta_start + ( beta_end - beta_start ) * ( i / ( t_max_steps - 1 ) )

def f_alpha(i):
    return 1 - f_beta(i)

def f_x(x0, t):
    alpha_values = [ f_alpha( i ) for i in range( t ) ]
    alpha_prod = math.prod( alpha_values )
    eps = np.random.randn(*x0.shape)
    term1 = math.sqrt( alpha_prod ) * x0
    term2 = math.sqrt( 1 - alpha_prod ) * eps
    x_t = term1 + term2
    return x_t

def visualize_noise():
    for t in range(t_max_steps):
        gauss_circle = np.zeros(scaled_circles_x.shape)
        for i in range(scaled_circles_x.shape[0]):
            gauss_circle[i] = f_x( scaled_circles_x[i], i )
        
        visualize(gauss_circle, noisy_circles_y)

class TinyDiffusionAutocoder(nn.Module):
    def __init__( self ):
        super(TinyDiffusionAutocoder, self).__init__()
        self.embed  = nn.Embedding(t_max_steps, 8)
        self.ln1    = nn.Linear(2 + 8, 64)
        self.ln12_a = nn.ReLU()
        self.ln2    = nn.Linear(64, 64)
        self.ln23_a = nn.ReLU()
        self.ln3    = nn.Linear(64, 2)

    def forward( self, x ):
        r, t = x
        em = self.embed(t)
        r = torch.cat([r, em], dim=-1)
        r = self.ln12_a( self.ln1( r ) )
        r = self.ln23_a( self.ln2( r ) )
        r = self.ln3( r )
        return r


model               = TinyDiffusionAutocoder()
circle_tensor       = torch.from_numpy( scaled_circles_x ).float()
beta_tensor         = torch.tensor( [ f_beta( i ) for i in range( t_max_steps ) ], dtype=torch.float32 )
alpha_tensor        = torch.tensor( [ f_alpha( i ) for i in range( t_max_steps ) ], dtype=torch.float32 )
alpha_prod_tensor   = torch.cumprod( alpha_tensor, dim=0, dtype=float )

optimizer           = optim.Adam(model.parameters(), lr=0.001)
loss_function       = nn.MSELoss()

batch_size = 500
assert circle_tensor.shape[0] % batch_size == 0

# Convention: the runner looks for a class named `Model` in each project file.
Model = TinyDiffusionAutocoder

@torch.no_grad()
def sample_reverse(net, num_samples, return_frames=False):
    net.eval()
    x_t = torch.randn(num_samples, 2)
    frames = [x_t.tolist()] if return_frames else None

    for t in reversed(range(t_max_steps)):
        t_en = torch.full((num_samples,), t, dtype=torch.long)
        eps_hat = net( (x_t, t_en) )

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

        if return_frames:
            frames.append(x_t.tolist())

    if return_frames:
        return frames
    return x_t


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    for epoch in range(1000):
        model.train()
        training_loss = 0
        training_count = 0
        for batch_index in range( 0, circle_tensor.shape[0], batch_size ):
            optimizer.zero_grad()
            circles = circle_tensor[batch_index:batch_index+batch_size]
            e_noise = torch.randn(batch_size, 2)
            t_en = torch.randint( low=0,high=t_max_steps,size=(batch_size,), dtype=torch.long)
            alpha_bar_t = alpha_prod_tensor[t_en]          # shape: (B,)
            alpha_bar_t = alpha_bar_t.unsqueeze(-1) # shape: (B, 1) for broadcasting
            term1 = torch.sqrt( alpha_bar_t ) * circles
            term2 = torch.sqrt( 1 - alpha_bar_t ) * e_noise
            x_t = (term1 + term2).float()
            input = ( x_t, t_en )
            preds = model( input )
            loss = loss_function(preds, e_noise)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            training_count += 1

        print(f"Epoch={epoch} | Loss={training_loss / ( training_count )}")

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "model.pt"))

    generated = sample_reverse(model, samples).numpy()

    plt.figure(figsize=(7, 7))
    plt.scatter(scaled_circles_x[:, 0], scaled_circles_x[:, 1], c='green', alpha=0.4, label='Real Data')
    plt.scatter(generated[:, 0], generated[:, 1], c='blue', alpha=0.6, label='Generated (Reverse Diffusion)')
    plt.title("Real vs Generated Points")
    plt.xlabel("Feature 1 (X coordinate)")
    plt.ylabel("Feature 2 (Y coordinate)")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.show()
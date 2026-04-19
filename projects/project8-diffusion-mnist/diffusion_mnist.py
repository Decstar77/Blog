import torch 
import torch.nn as nn
import torch.optim as optim
import random
import math
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
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
t_max_steps = 1000
torch.manual_seed(23)
random.seed(32)

def f_beta(i): 
    return beta_start + ( beta_end - beta_start ) * ( i / ( t_max_steps - 1 ) )

def f_alpha(i):
    return 1 - f_beta(i)

class TinyDiffusion(nn.Module):
    def __init__( self ):
        super(TinyDiffusion, self).__init__()
        
        chanels = [64, 128, 256]

        self.t_embed = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.l_embed = nn.Embedding(10, 32)

        self.ecn1    = nn.Conv2d( in_channels=1, out_channels=chanels[0], kernel_size=3, padding=1 )
        self.ecn1p   = nn.Linear(32, chanels[0] * 2)
        self.enc1n   = nn.GroupNorm(8, chanels[0])
        self.enc1a   = nn.ReLU()
        self.ecn1max = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ecn2    = nn.Conv2d(in_channels=chanels[0], out_channels=chanels[1], kernel_size=3, padding=1 )
        self.ecn2p   = nn.Linear(32, chanels[1] * 2)
        self.enc2n   = nn.GroupNorm(8, chanels[1])
        self.enc2a   = nn.ReLU()
        self.ecn2max = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ecn3    = nn.Conv2d(in_channels=chanels[1], out_channels=chanels[2], kernel_size=3, padding=1 )
        self.ecn3p   = nn.Linear(32, chanels[2] * 2)
        self.enc3n   = nn.GroupNorm(8, chanels[2])
        self.enc3a   = nn.ReLU()
        self.ecn3max = nn.MaxPool2d(kernel_size=2, stride=2)

        self.botcv  = nn.Conv2d(in_channels=chanels[2], out_channels=chanels[2], kernel_size=3, padding=1)
        self.botav  = nn.ReLU()

        self.up3     = nn.Upsample( scale_factor=2, mode='bilinear', align_corners=False )
        self.dcn3p   = nn.Linear(32, chanels[1] * 2 )
        self.dcn3    = nn.Conv2d( in_channels=chanels[2] * 2, out_channels=chanels[1], kernel_size=3, padding=1 )
        self.dcn3n   = nn.GroupNorm(8, chanels[1])
        self.dcn3a   = nn.ReLU()

        self.up2     = nn.Upsample( scale_factor=2, mode='bilinear', align_corners=False )
        self.dcn2p   = nn.Linear(32, chanels[0] * 2 )
        self.dcn2    = nn.Conv2d( in_channels=chanels[1] * 2, out_channels=chanels[0], kernel_size=3, padding=1 )
        self.dcn2n   = nn.GroupNorm(8, chanels[0])
        self.dcn2a   = nn.ReLU()

        self.up1     = nn.Upsample( scale_factor=2, mode='bilinear', align_corners=False )
        self.dcn1    = nn.Conv2d( in_channels=chanels[0] * 2, out_channels=1, kernel_size=3, padding=1 )

    def sinusoidal(self, t, d_model ):
        N = 10_000
        D = d_model
        K = torch.arange(0, D//2)

        positions = t.unsqueeze(1).float()
        embeddings = torch.zeros(t.shape[0], D)
        denominators = torch.pow( N, 2 * K / D ) 
        embeddings[:, 0::2] = torch.sin(positions/denominators)
        embeddings[:, 1::2] = torch.cos(positions/denominators)
        return embeddings

    # feature-wise linear modulation
    def film(self, linear, cond):
        gamma, beta = linear(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma, beta

    def forward( self, x ):
        x, t, l = x
        t_embed = self.t_embed( self.sinusoidal(t, 32) )
        l_embed = self.l_embed(l)
        cond = t_embed + l_embed

        r = self.ecn1(x)
        r = self.enc1n(r)
        gamma, beta = self.film(self.ecn1p, cond)
        s1 = self.enc1a( gamma * r + beta )
        r = self.ecn1max(s1)

        r = self.ecn2(r)
        r = self.enc2n(r)
        gamma, beta = self.film(self.ecn2p, cond)
        s2 = self.enc2a( gamma * r + beta )
        r = self.ecn2max(s2)
        
        r = self.ecn3(r)
        r = self.enc3n(r)
        gamma, beta = self.film(self.ecn3p, cond)
        s3 = self.enc3a( gamma * r + beta )
        r = self.ecn3max(s3)

        r = self.botav( self.botcv( r ) )

        r = self.up3(r)
        if r.shape[-2:] != s3.shape[-2:]:
            r = torch.nn.functional.interpolate(r, size=s3.shape[-2:], mode='bilinear', align_corners=False)

        r = torch.cat([r, s3], dim=1)
        r = self.dcn3( r )
        r = self.dcn3n( r )
        gamma, beta = self.film(self.dcn3p, cond)
        r = self.dcn3a( gamma * r + beta )
        
        r = self.up2(r)
        r = torch.cat([r, s2], dim=1)
        r = self.dcn2( r )
        r = self.dcn2n( r )
        gamma, beta = self.film(self.dcn2p, cond)
        r = self.dcn2a( gamma * r + beta )

        r = self.up1(r)
        r = torch.cat([r, s1], dim=1)
        r = self.dcn1( r )

        return r

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("No cuda using CPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_loader     = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
validation_loader   = DataLoader(mnist_validation, batch_size=batch_size, shuffle=True)
beta_tensor         = torch.tensor( [ f_beta( i ) for i in range( t_max_steps ) ], dtype=torch.float32 )
alpha_tensor        = torch.tensor( [ f_alpha( i ) for i in range( t_max_steps ) ], dtype=torch.float32 )
alpha_prod_tensor   = torch.cumprod( alpha_tensor, dim=0 )

model           = TinyDiffusion().to(device)
loss_function   = nn.MSELoss()
optimizer       = optim.Adam(model.parameters(), lr=1e-4)
scheduler       = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

#state_dict = torch.load("./projects/project8-diffusion-mnist/model.pt", weights_only=True)
#model.load_state_dict(state_dict)

for epoch in range(20):
    print(f"==================={epoch}===================")
    model.train()
    training_loss = 0
    training_count = 0
    pbar = tqdm(training_loader, desc="Training")
    for i, (x, y) in  enumerate(pbar):
        B = x.shape[0]
        optimizer.zero_grad()

        e_noise = torch.randn_like(x) # [64, 1, 28, 28]
        t_en = torch.randint(low=0, high=t_max_steps, size=(B,), dtype=torch.long)
        alpha_bar_t = alpha_prod_tensor[t_en].float().view(B, 1, 1, 1)  # [64, 1, 1, 1]
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * e_noise
        preds = model((x_t, t_en, y))
        loss = loss_function(preds, e_noise)
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
            t_en = torch.randint(low=0, high=t_max_steps, size=(B,), dtype=torch.long)
            alpha_bar_t = alpha_prod_tensor[t_en].float().view(B, 1, 1, 1)  # [64, 1, 1, 1]
            x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * e_noise
            preds = model((x_t, t_en, y))
            loss = loss_function(preds, e_noise)
            validation_loss+= loss
            validation_count+=1
            loss_str = f"{(validation_loss / validation_count):.5f}"
            pbar.set_postfix({"loss":loss_str})

    torch.save(model.state_dict(), "./projects/project8-diffusion-mnist/model.pt")

@torch.no_grad()
def sample_reverse(num_samples, value):
    model.eval()
    x_t = torch.randn(num_samples, 1, 28, 28)

    for t in reversed(range(t_max_steps)):
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

num_digits = 10
num_per_digit = 10
fig, axes = plt.subplots(num_per_digit, num_digits, figsize=(num_digits * 2, num_per_digit * 2))

for d in tqdm(range(num_digits), desc="Generating samples"):
    samples = sample_reverse(num_per_digit, d)
    for r in range(num_per_digit):
        image = samples[r].reshape(28, 28)
        image = (image + torch.ones(28, 28)) / 2
        axes[r][d].imshow(image, cmap='gray')
        axes[r][d].axis('off')

plt.tight_layout()
print("saving image")
plt.savefig('./projects/project8-diffusion-mnist/sample.png')
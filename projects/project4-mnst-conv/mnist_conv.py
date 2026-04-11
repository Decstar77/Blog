import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

train_dataset       = datasets.MNIST( root='./data/mnist/data', train=True,  download=True, transform=transform )
validation_dataset  = datasets.MNIST( root='./data/mnist/data', train=False, download=True, transform=transform )

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        #  H - (kernel-1)
        self.conv1 = nn.Conv2d(1, 32, 3) # -> 26, 26
        self.conv1_act = nn.ReLU()
        self.conv1_pool = nn.MaxPool2d(kernel_size=2) # -> 13, 13
        self.conv2 = nn.Conv2d(32, 64, 3) # -> 11, 11
        self.conv2_act = nn.ReLU()
        self.conv2_pool = nn.MaxPool2d(kernel_size=2) # -> 5,5
        self.output = nn.Linear(1600, 10) # [64, 5, 5]

    def forward(self, x):
        c1 = self.conv1_pool( self.conv1_act( self.conv1(x) ) )
        c2 = self.conv2_pool( self.conv2_act( self.conv2(c1) ) )
        f = torch.flatten(c2, 1 )
        r = self.output( f )
        return r

training_dataloader     = DataLoader( train_dataset,        64, shuffle=True)
validation_dataloader   = DataLoader( validation_dataset,   64 )

model       = Conv()
loss        = nn.CrossEntropyLoss()
optimizer   = optim.Adam( model.parameters(), lr=0.001 )

for epoch in range(100):
    for i, (training_inputs, training_labels) in enumerate(training_dataloader):
        optimizer.zero_grad()
        training_preds = model(training_inputs)
        training_loss = loss(training_preds, training_labels)
        training_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            with torch.no_grad():
                validation_inputs, validation_lables = next( iter(validation_dataloader) )
                validation_preds = model(validation_inputs)
                validation_loss = loss(validation_preds, validation_lables)
                print(f"train_loss={training_loss:.4f} | val_loss={validation_loss:.4f}")







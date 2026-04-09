import torch 
import torch.nn as nn
import torch.optim as optim

# Things I learnt
#   - Torch primitives
#   - Keep values small ~[-1, 1]
#   - MSE Regession problems
#   - Adam optimizer.

import random

def parabola( x ):
    return x ** 2

samples = 2000
training_x =  [ random.uniform(-1, 1) for _ in range(samples) ] 
random.shuffle( training_x ) 
training_y = [ parabola(x) for x in training_x ]

training_x_tensor = torch.tensor( training_x[:int(0.8 * samples)], dtype=torch.float32 ).reshape(-1, 1)
training_y_tensor = torch.tensor( training_y[:int(0.8 * samples)], dtype=torch.float32 )

validation_x_tensor = torch.tensor( training_x[int(0.8 * samples):], dtype=torch.float32 ).reshape(-1, 1)
validation_y_tensor = torch.tensor( training_y[int(0.8 * samples):], dtype=torch.float32 )

class FirstNN(nn.Module):
    def __init__(self):
        super(FirstNN, self).__init__()
        self.input = nn.Linear(1, 32)
        self.input_act = nn.Tanh()
        self.hidden = nn.Linear(32, 32)
        self.hidden_act = nn.Tanh()
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        l1 = self.input_act(self.input(x))
        l2 = self.hidden_act(self.hidden(l1))
        l3 = self.output(l2)
        return l3

model = FirstNN()
loss = nn.MSELoss()
optimizer = optim.Adam( model.parameters(), lr=0.001 )

for epoch in range(1000):
    optimizer.zero_grad()
    preds = model( training_x_tensor ).reshape(-1)
    train_loss = loss(preds, training_y_tensor)
    train_loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        with torch.no_grad():
            val_preds = model( validation_x_tensor ).reshape(-1)
            val_loss = loss(val_preds, validation_y_tensor )
            print(f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")



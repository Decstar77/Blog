import torch 
import torch.nn as nn
import torch.optim as optim
import random

from sklearn.datasets import make_circles

samples = 2000
noisy_circles_x, noisy_circles_y = make_circles( n_samples=samples, noise=0.1, random_state=6)

training_tensor_x = torch.tensor(noisy_circles_x[:int(0.8*samples)], dtype=torch.float32)
training_tensor_y = torch.tensor(noisy_circles_y[:int(0.8*samples)], dtype=torch.float32)

validation_tensor_x = torch.tensor(noisy_circles_x[int(0.8*samples):], dtype=torch.float32)
validation_tensor_y = torch.tensor(noisy_circles_y[int(0.8*samples):], dtype=torch.float32)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        size = 64
        self.input = nn.Linear(2, size)
        self.input_act = nn.Tanh()
        self.hidden = nn.Linear(size, size)
        self.hidden_act = nn.Tanh()
        self.output = nn.Linear(size, 1)

    def forward(self, x):
        res_1 = self.input_act( self.input(x) )
        res_2 = self.hidden_act( self.hidden(res_1) )
        res_3 = self.output(res_2)
        return res_3

model       = Classifier()
loss        = nn.BCEWithLogitsLoss()
optimizer   = optim.AdamW(model.parameters(), lr=0.001)

sigmoid = nn.Sigmoid()
def cacl_accuracy(predictions, answers):
    probabilities = sigmoid(predictions)
    guesses = torch.round(probabilities)
    total = torch.numel(guesses) 
    num_correct = total - torch.count_nonzero( guesses - answers )
    return num_correct / total

    
for epoch in range(1000):
    optimizer.zero_grad()
    model.train()
    outputResults = model(training_tensor_x)
    outputResults = outputResults.reshape(-1)
    trainingLoss = loss(outputResults, training_tensor_y)
    trainingLoss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            model.eval()
            validationResults = model(validation_tensor_x)
            validationResults = validationResults.reshape(-1)
            valLoss = loss( validationResults, validation_tensor_y )
            training_loss_f = float(trainingLoss.item())
            validation_loss_f = float(valLoss.item())
            validation_acc = cacl_accuracy(validationResults, validation_tensor_y)
            print(f"Training loss={training_loss_f:.4f} | Validation loss={validation_loss_f:.4f} | Validation acc = {validation_acc:.4f} ")
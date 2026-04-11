import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        torchvision.transforms.Lambda( lambda x: torch.flatten(x) )
    ])

mnist_train = torchvision.datasets.MNIST(root='./data/mnist/data', train=True, download=True, transform=transform)
mnist_validation  = torchvision.datasets.MNIST(root='./data/mnist/data', train=False, download=True, transform=transform)
batch_size = 64
training_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(mnist_validation, batch_size=batch_size, shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input = nn.Linear(784, 784)
        self.input_act = nn.Tanh()
        self.hidden = nn.Linear(784, 784)
        self.hidden_act = nn.Tanh()
        self.output = nn.Linear(784, 10)

    def forward(self, x):
        l1 = self.input_act(self.input(x))
        l2 = self.hidden_act(self.hidden(l1))
        l3 = self.output(l2)
        return l3

model       = MLP()
loss        = nn.CrossEntropyLoss()
optimizer   = optim.AdamW(model.parameters(), lr=0.001)

def cacl_accuracy(predictions, answers):
    pred_class = predictions.argmax(dim=1)
    equal = torch.eq(pred_class, answers).float()
    mean = torch.mean(equal)
    return mean

for epoch in range(1):
    for i, (inputs, labels) in enumerate(training_loader):
        optimizer.zero_grad()
        model.train()
        logits = model(inputs)
        training_loss = loss(logits, labels)
        training_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                val_iter = iter(validation_loader)
                val_inputs, val_labels = next(val_iter)
                val_logits = model(val_inputs)
                val_loss = loss(val_logits, val_labels)
                val_acc = cacl_accuracy(val_logits, val_labels)
                print(f"t_loss={training_loss:.4f}|v_loss={val_loss:.4f}|v_acc={val_acc:.4f}")

    with torch.no_grad():
        model.eval()
        avg_loss = 0
        avg_acc = 0
        for j, (val_inputs, val_labels) in enumerate(validation_loader):
            val_logits = model(val_inputs)
            val_loss = loss(val_logits, val_labels)
            val_acc = cacl_accuracy(val_logits, val_labels)
            avg_loss += val_loss

        print(f"EPOC v_loss={val_loss:.4f}|v_acc={val_acc:.4f}")
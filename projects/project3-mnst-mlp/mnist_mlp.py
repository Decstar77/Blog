import torch
import torch.nn as nn

# Convention: the runner looks for a class named `Model` in each project file.
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input      = nn.Linear(784, 784)
        self.input_act  = nn.Tanh()
        self.hidden     = nn.Linear(784, 784)
        self.hidden_act = nn.Tanh()
        self.output     = nn.Linear(784, 10)

    def forward(self, x):
        l1 = self.input_act(self.input(x))
        l2 = self.hidden_act(self.hidden(l1))
        return self.output(l2)


if __name__ == "__main__":
    import torch.optim as optim
    import torchvision
    from torch.utils.data import DataLoader
    import os

    # Things I learnt:
    #   - Multi-class classification with CrossEntropyLoss
    #   - DataLoader + batching
    #   - Argmax for class prediction

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
    ])

    train_data = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    val_data   = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=64)

    model     = Model()
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    def accuracy(logits, labels):
        return (logits.argmax(dim=1) == labels).float().mean().item()

    for epoch in range(10):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        total_acc, n = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                total_acc += accuracy(model(inputs), labels) * len(labels)
                n += len(labels)
        print(f"epoch={epoch+1:2d}  val_acc={total_acc/n:.4f}")

    out = os.path.join(os.path.dirname(__file__), "model.pt")
    torch.save(model.state_dict(), out)
    print(f"Saved → {out}")

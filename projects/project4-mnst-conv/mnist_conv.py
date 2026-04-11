import torch
import torch.nn as nn

# Convention: the runner looks for a class named `Model` in each project file.
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 28x28 → conv(k=3) → 26x26 → maxpool(2) → 13x13
        self.conv1      = nn.Conv2d(1, 32, 3)
        self.conv1_act  = nn.ReLU()
        self.conv1_pool = nn.MaxPool2d(2)
        # 13x13 → conv(k=3) → 11x11 → maxpool(2) → 5x5
        self.conv2      = nn.Conv2d(32, 64, 3)
        self.conv2_act  = nn.ReLU()
        self.conv2_pool = nn.MaxPool2d(2)
        # 64 × 5 × 5 = 1600
        self.output     = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.conv1_pool(self.conv1_act(self.conv1(x)))
        x = self.conv2_pool(self.conv2_act(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.output(x)


if __name__ == "__main__":
    import torch.optim as optim
    import torchvision
    from torch.utils.data import DataLoader
    import os

    # Things I learnt:
    #   - Convolutional layers + spatial feature extraction
    #   - MaxPooling for downsampling
    #   - CNNs are parameter-efficient vs large MLPs

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ])

    train_data = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    val_data   = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=64)

    model     = Model()
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def accuracy(logits, labels):
        return (logits.argmax(dim=1) == labels).float().mean().item()

    for epoch in range(5):
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

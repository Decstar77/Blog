import torch
import torch.nn as nn

# Convention: the runner looks for a class named `Model` in each project file.
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        size = 64
        self.input      = nn.Linear(2, size)
        self.input_act  = nn.Tanh()
        self.hidden     = nn.Linear(size, size)
        self.hidden_act = nn.Tanh()
        self.output     = nn.Linear(size, 1)

    def forward(self, x):
        x = self.input_act(self.input(x))
        x = self.hidden_act(self.hidden(x))
        return self.output(x)


if __name__ == "__main__":
    import torch.optim as optim
    import os
    from sklearn.datasets import make_circles

    # Things I learnt:
    #   - Binary classification with BCEWithLogitsLoss
    #   - AdamW optimizer
    #   - Non-linearly separable data

    samples = 2000
    xs, ys = make_circles(n_samples=samples, noise=0.1, random_state=6)

    split = int(0.8 * samples)
    train_x = torch.tensor(xs[:split], dtype=torch.float32)
    train_y = torch.tensor(ys[:split], dtype=torch.float32)
    val_x   = torch.tensor(xs[split:], dtype=torch.float32)
    val_y   = torch.tensor(ys[split:], dtype=torch.float32)

    model     = Model()
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    sigmoid   = nn.Sigmoid()

    def accuracy(logits, labels):
        probs    = sigmoid(logits)
        guesses  = torch.round(probs)
        correct  = torch.numel(guesses) - torch.count_nonzero(guesses - labels)
        return float(correct / torch.numel(guesses))

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        logits     = model(train_x).reshape(-1)
        train_loss = loss_fn(logits, train_y)
        train_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_x).reshape(-1)
                val_loss   = loss_fn(val_logits, val_y)
                val_acc    = accuracy(val_logits, val_y)
                print(f"epoch={epoch:4d}  train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.4f}")

    out = os.path.join(os.path.dirname(__file__), "model.pt")
    torch.save(model.state_dict(), out)
    print(f"Saved → {out}")

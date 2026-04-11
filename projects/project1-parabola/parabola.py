import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# Convention: the runner looks for a class named `Model` in each project file.
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input      = nn.Linear(1, 32)
        self.input_act  = nn.Tanh()
        self.hidden     = nn.Linear(32, 32)
        self.hidden_act = nn.Tanh()
        self.output     = nn.Linear(32, 1)

    def forward(self, x):
        l1 = self.input_act(self.input(x))
        l2 = self.hidden_act(self.hidden(l1))
        l3 = self.output(l2)
        return l3


if __name__ == "__main__":
    # Things I learnt:
    #   - Torch primitives
    #   - Keep values small ~[-1, 1]
    #   - MSE regression problems
    #   - Adam optimizer

    def parabola(x):
        return x ** 2

    samples = 2000
    training_x = [random.uniform(-1, 1) for _ in range(samples)]
    random.shuffle(training_x)
    training_y = [parabola(x) for x in training_x]

    split = int(0.8 * samples)

    training_x_tensor   = torch.tensor(training_x[:split], dtype=torch.float32).reshape(-1, 1)
    training_y_tensor   = torch.tensor(training_y[:split], dtype=torch.float32)
    validation_x_tensor = torch.tensor(training_x[split:], dtype=torch.float32).reshape(-1, 1)
    validation_y_tensor = torch.tensor(training_y[split:], dtype=torch.float32)

    model     = Model()
    loss_fn   = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):
        optimizer.zero_grad()
        preds      = model(training_x_tensor).reshape(-1)
        train_loss = loss_fn(preds, training_y_tensor)
        train_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                val_preds = model(validation_x_tensor).reshape(-1)
                val_loss  = loss_fn(val_preds, validation_y_tensor)
                print(f"epoch={epoch:4d}  train={train_loss:.6f}  val={val_loss:.6f}")

    out = os.path.join(os.path.dirname(__file__), "model.pt")
    torch.save(model.state_dict(), out)
    print(f"Saved → {out}")

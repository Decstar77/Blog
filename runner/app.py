import random
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model ────────────────────────────────────────────────────────

class ParabolaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input  = nn.Linear(1, 32)
        self.act1   = nn.Tanh()
        self.hidden = nn.Linear(32, 32)
        self.act2   = nn.Tanh()
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.act1(self.input(x))
        x = self.act2(self.hidden(x))
        return self.output(x)


# ── Train on startup ─────────────────────────────────────────────

print("Training parabola model...")

samples = 2000
xs = [random.uniform(-1, 1) for _ in range(samples)]
ys = [x ** 2 for x in xs]

split = int(0.8 * samples)
train_x = torch.tensor(xs[:split], dtype=torch.float32).reshape(-1, 1)
train_y = torch.tensor(ys[:split], dtype=torch.float32)

model = ParabolaNet()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    preds = model(train_x).reshape(-1)
    loss = loss_fn(preds, train_y)
    loss.backward()
    optimizer.step()

model.eval()
print(f"Training complete. Final loss: {loss.item():.6f}")


# ── API ──────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    x: float


@app.post("/run/parabola")
def run_parabola(req: RunRequest):
    x = max(-1.0, min(1.0, req.x))
    with torch.no_grad():
        x_tensor = torch.tensor([[x]], dtype=torch.float32)
        prediction = float(model(x_tensor).item())
    actual = x ** 2
    return {
        "prediction": round(prediction, 6),
        "actual":     round(actual, 6),
        "error":      round(abs(prediction - actual), 6),
    }


@app.get("/health")
def health():
    return {"status": "ok"}

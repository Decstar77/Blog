import importlib.util
import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.datasets import make_circles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Project loader ────────────────────────────────────────────────
# Each project file must define a class named `Model` and ship a
# `model.pt` state dict next to it.  The runner imports the file
# without executing anything (training lives behind __main__ guard).

def load_project(py_path: str, weights_path: str):
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Project file not found: {py_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}  — run the project script first")

    spec   = importlib.util.spec_from_file_location("project_module", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Model"):
        raise AttributeError(f"{py_path} must define a class named `Model`")

    model = module.Model()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    return model


# ── Load models at startup ────────────────────────────────────────

BASE = os.path.dirname(__file__)

parabola_model = load_project(
    py_path      = os.path.join(BASE, "projects/project1-parabola/parabola.py"),
    weights_path = os.path.join(BASE, "projects/project1-parabola/model.pt"),
)
print("Parabola model loaded.")

circle_model = load_project(
    py_path      = os.path.join(BASE, "projects/project2-classifier/circle_classifier.py"),
    weights_path = os.path.join(BASE, "projects/project2-classifier/model.pt"),
)
print("Circle classifier model loaded.")

# Pre-generate the circle dataset (deterministic — random_state=6)
_circle_xs, _circle_ys = make_circles(n_samples=2000, noise=0.1, random_state=6)
circle_dataset = {
    "points": [{"x": float(p[0]), "y": float(p[1]), "label": int(l)}
               for p, l in zip(_circle_xs, _circle_ys)]
}


# ── API ──────────────────────────────────────────────────────────

# ── Parabola ─────────────────────────────────────────────────────

class ParabolaRequest(BaseModel):
    x: float


@app.post("/run/parabola")
def run_parabola(req: ParabolaRequest):
    x = max(-1.0, min(1.0, req.x))
    with torch.no_grad():
        prediction = float(parabola_model(torch.tensor([[x]], dtype=torch.float32)).item())
    actual = x ** 2
    return {
        "prediction": round(prediction, 6),
        "actual":     round(actual, 6),
        "error":      round(abs(prediction - actual), 6),
    }


# ── Circle classifier ─────────────────────────────────────────────

class CirclePointRequest(BaseModel):
    x: float
    y: float

class CircleGridRequest(BaseModel):
    resolution: int = 50


@app.post("/run/circle-classifier")
def run_circle(req: CirclePointRequest):
    with torch.no_grad():
        t     = torch.tensor([[req.x, req.y]], dtype=torch.float32)
        logit = float(circle_model(t).item())
        prob  = float(torch.sigmoid(torch.tensor(logit)).item())
    return {
        "probability": round(prob, 6),
        "label":       1 if prob >= 0.5 else 0,
    }


@app.post("/run/circle-classifier/grid")
def run_circle_grid(req: CircleGridRequest):
    n = max(10, min(req.resolution, 100))
    r = 1.5  # coordinate range [-1.5, 1.5]

    # Build grid: row gy = top-to-bottom (y decreases), col gx = left-to-right (x increases)
    # This matches canvas pixel order so the JS can render without flipping.
    points = []
    for gy in range(n):
        for gx in range(n):
            px = -r + (gx + 0.5) / n * (2 * r)
            py =  r - (gy + 0.5) / n * (2 * r)
            points.append([px, py])

    tensor = torch.tensor(points, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(circle_model(tensor).reshape(-1)).tolist()

    grid = [probs[gy * n:(gy + 1) * n] for gy in range(n)]
    return {"grid": grid, "range": [-r, r], "resolution": n}


@app.get("/data/circle-classifier")
def circle_data():
    return circle_dataset


# ── Health ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

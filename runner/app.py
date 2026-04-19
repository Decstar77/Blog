import importlib.util
import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.datasets import make_circles
from typing import List

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

def load_project(py_path: str, weights_path: str, return_module: bool = False):
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Project file not found: {py_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}  — run the project script first")

    # Use the filename as the module name so multiple projects don't collide in sys.modules
    module_name = os.path.splitext(os.path.basename(py_path))[0]
    spec   = importlib.util.spec_from_file_location(module_name, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Model"):
        raise AttributeError(f"{py_path} must define a class named `Model`")

    model = module.Model()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    if return_module:
        return model, module
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
print("Circle classifier loaded.")

mnist_mlp_model = load_project(
    py_path      = os.path.join(BASE, "projects/project3-mnst-mlp/mnist_mlp.py"),
    weights_path = os.path.join(BASE, "projects/project3-mnst-mlp/model.pt"),
)
print("MNIST MLP loaded.")

mnist_conv_model = load_project(
    py_path      = os.path.join(BASE, "projects/project4-mnst-conv/mnist_conv.py"),
    weights_path = os.path.join(BASE, "projects/project4-mnst-conv/model.pt"),
)
print("MNIST CNN loaded.")

diffusion_model, diffusion_module = load_project(
    py_path       = os.path.join(BASE, "projects/project7-diffusion-circle/diffusion_circle.py"),
    weights_path  = os.path.join(BASE, "projects/project7-diffusion-circle/model.pt"),
    return_module = True,
)
print("Diffusion (circle) loaded.")

# Pre-generate the circle dataset (deterministic — random_state=6)
_circle_xs, _circle_ys = make_circles(n_samples=2000, noise=0.1, random_state=6)
circle_dataset = {
    "points": [{"x": float(p[0]), "y": float(p[1]), "label": int(l)}
               for p, l in zip(_circle_xs, _circle_ys)]
}

# Pre-extract the standardised circles dataset used by the diffusion project
diffusion_dataset = {
    "points": [{"x": float(p[0]), "y": float(p[1]), "label": int(l)}
               for p, l in zip(diffusion_module.scaled_circles_x.tolist(),
                               diffusion_module.noisy_circles_y.tolist())]
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
    r = 1.5

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


# ── MNIST ─────────────────────────────────────────────────────────

class MnistRequest(BaseModel):
    pixels: List[float]  # 784 floats, normalised to [-1, 1], row-major


@app.post("/run/mnist")
def run_mnist(req: MnistRequest):
    if len(req.pixels) != 784:
        return {"error": "Expected 784 pixels"}

    pixels = torch.tensor(req.pixels, dtype=torch.float32)

    with torch.no_grad():
        mlp_logits  = mnist_mlp_model(pixels.unsqueeze(0)).squeeze(0)       # (10,)
        cnn_logits  = mnist_conv_model(pixels.reshape(1, 1, 28, 28)).squeeze(0)  # (10,)

    mlp_probs = torch.softmax(mlp_logits, dim=0).tolist()
    cnn_probs = torch.softmax(cnn_logits, dim=0).tolist()

    return {
        "mlp": {
            "prediction":    int(mlp_logits.argmax().item()),
            "probabilities": [round(p, 4) for p in mlp_probs],
        },
        "cnn": {
            "prediction":    int(cnn_logits.argmax().item()),
            "probabilities": [round(p, 4) for p in cnn_probs],
        },
    }


# ── Diffusion (circles) ───────────────────────────────────────────

class DiffusionRequest(BaseModel):
    num_samples: int = 600


@app.post("/run/diffusion-circle")
def run_diffusion_circle(req: DiffusionRequest):
    n = max(1, min(req.num_samples, 2000))
    frames = diffusion_module.sample_reverse(diffusion_model, n, return_frames=True)
    return {"frames": frames}


@app.get("/data/diffusion-circle")
def diffusion_data():
    return diffusion_dataset


# ── Health ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

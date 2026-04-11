import importlib.util
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


# ── API ──────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    x: float


@app.post("/run/parabola")
def run_parabola(req: RunRequest):
    x = max(-1.0, min(1.0, req.x))
    with torch.no_grad():
        x_tensor   = torch.tensor([[x]], dtype=torch.float32)
        prediction = float(parabola_model(x_tensor).item())
    actual = x ** 2
    return {
        "prediction": round(prediction, 6),
        "actual":     round(actual, 6),
        "error":      round(abs(prediction - actual), 6),
    }


@app.get("/health")
def health():
    return {"status": "ok"}

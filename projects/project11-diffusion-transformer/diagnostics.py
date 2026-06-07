"""
Diagnostics for the DiT in diffusion_transformer.py.

Reuses the model definition by importing it. Loads model.pt and runs a battery
of probes that surface the failure mode where the network ignores its
conditioning (t and class label) and just learns "predict the input as noise" —
which is what produced the blobby samples earlier.

Run:  python diagnostics.py
"""

import importlib.util
import math
import sys
from pathlib import Path

import torch


HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.pt"
SRC_PATH = HERE / "diffusion_transformer.py"


def load_model_module():
    """Import diffusion_transformer.py up to (but not including) DataLoader
    construction, so we get class definitions without triggering training."""
    src = SRC_PATH.read_text()
    cut = src.index("training_loader")
    snippet = src[:cut]
    ns = {"__name__": "_dit_defs"}
    exec(snippet, ns)
    return ns


def build_model(ns, device):
    Model = ns["Model"]
    model = Model().to(device)
    if MODEL_PATH.exists():
        sd = torch.load(MODEL_PATH, weights_only=True, map_location=device)
        model.load_state_dict(sd)
        print(f"loaded weights from {MODEL_PATH}")
    else:
        print(f"WARNING: {MODEL_PATH} not found; running on randomly-initialised model")
    model.eval()
    return model


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def weight_stats(model):
    """Healthy DiT after training has adaLN modulation weights with std on the
    order of 0.05–0.2. Values near 0 mean conditioning never escaped the
    zero-init attractor — see the bias-zeroing bug."""
    section("WEIGHT STATS  (small std on adal/proj => undertrained conditioning)")
    rows = []
    for name, p in model.named_parameters():
        if any(k in name for k in ("adal", "final.proj", "pos_embed",
                                    "time_mlp", "label_embed")):
            rows.append((name, tuple(p.shape), p.float().std().item()))
    width = max(len(r[0]) for r in rows)
    for name, shape, std in rows:
        flag = "  <-- suspiciously small" if "adal" in name and std < 0.02 else ""
        print(f"  {name:<{width}}  shape={str(shape):<18} std={std:.4f}{flag}")


def conditioning_response(model, device):
    """The two probes that diagnosed the original bug:

    1) Time-conditioning probe.  Run ε̂(x, t=0) and ε̂(x, t=999) on the SAME
       random x. A working DiT produces very different outputs at the two
       extremes (cosine sim well below 0.5). A model that learned to ignore t
       returns nearly identical outputs (cosine sim ≈ 1).

    2) Label-conditioning probe.  ε̂(x, y=0) vs ε̂(x, y=9). Mean abs diff should
       be on the order of the output std. If it's ~0, the class label is
       being ignored.
    """
    section("CONDITIONING RESPONSE  (does the model actually use t and y?)")
    torch.manual_seed(0)
    B = 8
    x = torch.randn(B, 1, 28, 28, device=device)

    t0 = torch.zeros(B, dtype=torch.long, device=device)
    t999 = torch.full((B,), 999, dtype=torch.long, device=device)
    y0 = torch.zeros(B, dtype=torch.long, device=device)
    y9 = torch.full((B,), 9, dtype=torch.long, device=device)

    with torch.no_grad():
        e_t0 = model((x, t0,   y0))
        e_t999 = model((x, t999, y0))
        e_y0 = model((x, t999, y0))
        e_y9 = model((x, t999, y9))

    def cos(a, b):
        a = a.flatten(); b = b.flatten()
        return (a @ b / (a.norm() * b.norm())).item()

    print(f"  cosine_sim( eps_hat at t=0, eps_hat at t=999 )  = {cos(e_t0, e_t999):.3f}")
    print(f"     ^ healthy: << 0.5    broken (ignores t):  ~ 1.0")
    print()
    print(f"  ||eps_hat(y=0) - eps_hat(y=9)|| / ||eps_hat||  = "
          f"{(e_y0 - e_y9).norm().item() / e_y0.norm().item():.3f}")
    print(f"  mean|eps_hat(y=0) - eps_hat(y=9)|              = "
          f"{(e_y0 - e_y9).abs().mean().item():.4f}")
    print(f"  output std for reference                       = "
          f"{e_y0.std().item():.4f}")
    print(f"     ^ healthy: relative diff ~ O(1).  broken (ignores y): ~ 0")


def predicts_input_as_noise(model, device):
    """At t = T-1, x_t IS approximately pure noise, so corr(eps_hat, x_t) ≈ 1
    is correct. At t = 0, x_t is a clean image and the *correct* eps_hat is
    nearly independent of x_t, so this correlation should drop substantially.
    A model stuck at the "predict input as noise" minimum keeps the
    correlation near 1 across all t."""
    section("EPS_HAT vs INPUT CORRELATION ACROSS TIMESTEPS")
    torch.manual_seed(1)
    B = 32
    x = torch.randn(B, 1, 28, 28, device=device)
    y = torch.zeros(B, dtype=torch.long, device=device)

    def corr(a, b):
        a = a.flatten().float(); b = b.flatten().float()
        a = a - a.mean(); b = b - b.mean()
        return (a @ b / (a.norm() * b.norm())).item()

    print("  t       corr(eps_hat, x_t)   notes")
    for t in [0, 100, 250, 500, 750, 999]:
        t_en = torch.full((B,), t, dtype=torch.long, device=device)
        with torch.no_grad():
            eh = model((x, t_en, y))
        c = corr(eh, x)
        note = ""
        if t == 0 and c > 0.5:
            note = "<-- broken: clean-image step shouldn't correlate with x"
        if t == 999 and c < 0.8:
            note = "<-- pure-noise step should correlate strongly"
        print(f"  {t:4d}    {c:+.3f}              {note}")


def output_magnitude(model, device):
    """Sanity check that the network produces outputs with the right scale
    (true ε is N(0, I), so output std should be near 1.0). Wildly off
    magnitudes would cause the DDPM reverse step to over- or under-correct."""
    section("OUTPUT MAGNITUDE  (eps target has std=1; predictions should match)")
    torch.manual_seed(2)
    B = 16
    x = torch.randn(B, 1, 28, 28, device=device)
    print("  t      mean         std          min       max")
    for t in [0, 250, 500, 999]:
        t_en = torch.full((B,), t, dtype=torch.long, device=device)
        y = torch.zeros(B, dtype=torch.long, device=device)
        with torch.no_grad():
            eh = model((x, t_en, y))
        print(f"  {t:4d}   {eh.mean().item():+.4f}      "
              f"{eh.std().item():.4f}       "
              f"{eh.min().item():+.3f}    {eh.max().item():+.3f}")


def per_t_loss(model, device, num_batches=4, batch_size=64):
    """The MSE loss averaged over t hides bad behaviour at the tails. Bucket
    the loss by timestep range. A model that ignores t will have *high* loss
    at small t (where the right answer is near zero but it predicts ~x) even
    if the average looks fine."""
    section("PER-TIMESTEP-BUCKET DENOISING LOSS  (on real MNIST)")
    try:
        import torchvision
        from torch.utils.data import DataLoader
        ds = torchvision.datasets.MNIST(
            root=str(HERE / "data"), train=False, download=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"  skipped — couldn't load MNIST: {e}")
        return

    # Re-derive the alpha schedule locally so this file is self-contained.
    tmax = 1000
    beta_start, beta_end = 1e-4, 0.02
    betas = torch.linspace(beta_start, beta_end, tmax, device=device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    buckets = [(0, 50), (50, 200), (200, 500), (500, 800), (800, 1000)]
    sums = {b: 0.0 for b in buckets}
    counts = {b: 0 for b in buckets}

    seen = 0
    for x0, y in loader:
        if seen >= num_batches:
            break
        seen += 1
        x0 = x0.to(device); y = y.to(device)
        B = x0.shape[0]
        for lo, hi in buckets:
            t_en = torch.randint(lo, hi, (B,), device=device)
            ab = alpha_bar[t_en].view(B, 1, 1, 1)
            eps = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            with torch.no_grad():
                eh = model((xt, t_en, y))
            sums[(lo, hi)] += ((eh - eps) ** 2).mean().item() * B
            counts[(lo, hi)] += B

    print("  t-range          MSE        notes")
    for b in buckets:
        mse = sums[b] / max(counts[b], 1)
        note = ""
        if b[0] == 0 and mse > 0.3:
            note = "<-- bad: model can't denoise nearly-clean images"
        print(f"  [{b[0]:3d},{b[1]:4d})       {mse:.4f}     {note}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    ns = load_model_module()
    model = build_model(ns, device)

    weight_stats(model)
    conditioning_response(model, device)
    predicts_input_as_noise(model, device)
    output_magnitude(model, device)
    per_t_loss(model, device)


if __name__ == "__main__":
    main()

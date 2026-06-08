"""Minimal GPU smoke test for CleanDiffuser (lightning branch)."""

import sys

import torch
from cleandiffuser.nn_diffusion.mlps import MlpNNDiffusion


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False")
        return 1

    device = torch.device("cuda:0")
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(device)}")

    model = MlpNNDiffusion(x_dim=10, emb_dim=16, hidden_dims=256).to(device)
    x = torch.randn(4, 10, device=device)
    t = torch.randint(0, 1000, (4,), device=device)
    condition = torch.randn(4, 16, device=device)

    with torch.no_grad():
        out = model(x, t, condition)

    assert out.shape == (4, 10), f"unexpected shape: {out.shape}"
    print(f"forward_ok shape={tuple(out.shape)} device={out.device}")
    print("GPU smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

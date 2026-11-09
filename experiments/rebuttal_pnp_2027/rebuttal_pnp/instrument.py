"""Non-invasive cost instrumentation for existing samplers.

* ``NetworkCounter`` attaches a forward hook to the denoiser network (e.g.
  ``agent.model_ema["diffusion"]``) and counts forward calls and the number of
  sample-forwards (batch elements). When a forward output requires grad, a tensor
  hook counts each backward pass through the network (VJP / Jacobian products).
* ``GradCounter`` wraps an instance's ``gradients`` method (objective-gradient
  evaluations) without touching the class.
* ``cuda_time`` measures wall-clock with device synchronization before and after.
"""

from __future__ import annotations

import contextlib
import time
from typing import Callable, Dict

import torch


class NetworkCounter:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.forward_calls = 0
        self.forward_samples = 0
        self.backward_calls = 0
        self.backward_samples = 0
        self._handle = None

    def _hook(self, _module, inputs, output):
        n = int(inputs[0].shape[0]) if inputs and torch.is_tensor(inputs[0]) else 0
        self.forward_calls += 1
        self.forward_samples += n
        if torch.is_tensor(output) and output.requires_grad:
            def _bwd(grad, n=n):
                self.backward_calls += 1
                self.backward_samples += n
                return grad

            output.register_hook(_bwd)

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def as_dict(self) -> Dict[str, int]:
        return {
            "net_forward_calls": self.forward_calls,
            "net_forward_samples": self.forward_samples,
            "net_backward_calls": self.backward_calls,
            "net_backward_samples": self.backward_samples,
        }


class GradCounter:
    """Count calls to ``obj.gradients`` (objective-gradient evaluations)."""

    def __init__(self, obj, method: str = "gradients"):
        self.obj = obj
        self.method = method
        self.calls = 0
        self.samples = 0
        self._orig = None

    def __enter__(self):
        self._had_instance_attr = self.method in vars(self.obj)
        self._orig = getattr(self.obj, self.method)
        orig = self._orig

        def wrapped(x, *args, **kwargs):
            self.calls += 1
            self.samples += int(x.shape[0])
            return orig(x, *args, **kwargs)

        setattr(self.obj, self.method, wrapped)
        return self

    def __exit__(self, *exc):
        if self._had_instance_attr:
            setattr(self.obj, self.method, self._orig)
        else:
            delattr(self.obj, self.method)  # class method becomes visible again
        return False

    def as_dict(self) -> Dict[str, int]:
        return {"grad_calls": self.calls, "grad_samples": self.samples}


def sync(device=None) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def cuda_time(out: Dict[str, float], key: str = "seconds"):
    sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        sync()
        out[key] = out.get(key, 0.0) + (time.perf_counter() - t0)


def warmup(fn: Callable[[], object], n: int = 3) -> None:
    for _ in range(n):
        fn()
    sync()

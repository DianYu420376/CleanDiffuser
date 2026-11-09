import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rebuttal_pnp import safe_io  # noqa: E402
from rebuttal_pnp.instrument import GradCounter, NetworkCounter  # noqa: E402


def test_safe_io_refuses_original_repo_and_outside_roots(tmp_path):
    with pytest.raises(PermissionError):
        safe_io.write_text(safe_io.ORIGINAL_REPO / "results" / "x.txt", "no")
    with pytest.raises(PermissionError):
        safe_io.write_text(tmp_path / "x.txt", "no")


def test_safe_io_no_overwrite_and_new_run_dir():
    parent = safe_io.RESULTS_ROOT / "_unit_test"
    run = safe_io.new_run_dir(parent, run_id=f"t_{torch.randint(0, 10**9, (1,)).item()}")
    try:
        p = safe_io.write_json(run / "a.json", {"x": 1})
        assert p.read_text().strip().startswith("{")
        with pytest.raises(safe_io.OutputExistsError):
            safe_io.write_json(run / "a.json", {"x": 2})
        assert '"x": 1' in p.read_text()
        with pytest.raises(safe_io.OutputExistsError):
            safe_io.new_run_dir(parent, run_id=run.name)
        assert not any(q.name.endswith(".tmp") for q in run.iterdir())
    finally:
        for q in run.iterdir():
            q.unlink()
        run.rmdir()


def test_network_counter_counts_forward_and_backward():
    net = torch.nn.Linear(3, 3)
    with NetworkCounter(net) as cnt:
        with torch.no_grad():
            net(torch.randn(5, 3))
        x = torch.randn(2, 3, requires_grad=True)
        y = net(x).sum()
        torch.autograd.grad(y, x)
    assert cnt.forward_calls == 2 and cnt.forward_samples == 7
    assert cnt.backward_calls == 1 and cnt.backward_samples == 2
    net(torch.randn(1, 3))
    assert cnt.forward_calls == 2  # hook removed


def test_grad_counter_wraps_and_restores():
    class C:
        def gradients(self, x, t, c=None):
            return None, x

    obj = C()
    with GradCounter(obj) as gc:
        obj.gradients(torch.zeros(4, 2), None)
        obj.gradients(torch.zeros(1, 2), None)
    assert gc.calls == 2 and gc.samples == 5
    assert "gradients" not in vars(obj)

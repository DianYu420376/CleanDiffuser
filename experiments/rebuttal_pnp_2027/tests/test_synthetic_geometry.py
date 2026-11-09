import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pipelines"))

from rebuttal_pnp.synthetic_geometry import SyntheticGeometry, project_ellipsoid  # noqa: E402

ORIG = Path("/u/rzhang26/CleanDiffuser/results/guidance_synthetic")


def test_ellipsoid_projection_kkt_and_optimality():
    rng = np.random.default_rng(0)
    a = np.array([1.0, 0.85, 0.7, 0.55, 0.4])
    u = rng.normal(size=(500, 5)) * 2.0
    p = project_ellipsoid(u, a)
    ns = np.sum((p / a) ** 2, axis=1)
    outside = np.sum((u / a) ** 2, axis=1) > 1
    assert np.all(ns <= 1 + 1e-9)
    assert np.allclose(ns[outside], 1.0, atol=1e-9)
    assert np.allclose(p[~outside], u[~outside])
    # no random feasible point is closer than the projection
    cand = rng.normal(size=(20000, 5))
    cand = cand / np.sqrt(np.sum((cand / a) ** 2, axis=1, keepdims=True)) * rng.uniform(0, 1, (20000, 1)) ** 0.2
    for i in np.where(outside)[0][:20]:
        assert np.linalg.norm(u[i] - p[i]) <= np.min(np.linalg.norm(cand - u[i], axis=1)) + 1e-9


def test_xstar_matches_existing_solver():
    import guidance_synthetic_subspace as gss

    for t in ("test_a", "test_b"):
        geo = gss._load_geometry(ORIG / t / "geometry.json")
        sg = SyntheticGeometry(geo)
        x_opt, _ = gss.solve_optimal_feasible(geo)
        assert np.linalg.norm(sg.x_star - x_opt) < 1e-4
        m = sg.metrics(sg.x_star[None])
        assert m["dist_K"][0] < 1e-9 and m["G_norm"][0] < 1e-6
        # existing violation metric reproduced verbatim
        x = np.random.default_rng(1).normal(size=(50, 10))
        assert np.allclose(sg.metrics(x)["violation_existing"], gss.constraint_violation(x, geo))
        assert np.allclose(sg.metrics(x)["objective"], -gss.reward_values(x, geo))
        # G at eta=1/L equals 2(x - x*)  ->  ||x - x*|| relation
        assert np.all(sg.metrics(x)["dist_K"] <= sg.metrics(x)["violation_existing"] + 1e-12)

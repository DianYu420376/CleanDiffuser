"""Exact geometry + per-iterate metrics for the synthetic subspace tests (A: box, B: ellipsoid).

All quantities are in PHYSICAL (unnormalized) ambient coordinates. The feasible set is
K = {B u : u in C} with B (10 x d) orthonormal columns and C a box [-w, w]^d (test A) or
the axis-aligned ellipsoid {u : sum (u_i/a_i)^2 <= 1} (test B), both centered at 0, so the
"data manifold" is the linear subspace span(B) through the origin.

* ``project_K`` is the exact Euclidean projection: P_K(x) = B P_C(B^T x). For the box,
  P_C is clipping (identical to the existing ``gss.project_to_feasible``). For the ellipsoid
  P_C solves the KKT system v_i = u_i a_i^2 / (a_i^2 + lam), lam >= 0 with
  sum (a_i u_i / (a_i^2 + lam))^2 = 1 by bisection on lam (float64). The existing
  ``gss.constraint_violation`` for test B uses *radial scaling*, which is not the Euclidean
  projection; we keep it for comparability and report ``dist_K`` separately.
* f(x) = ||x - c||^2, so the constrained minimizer is x* = P_K(c) and f* = ||P_K(c) - c||^2.
* Projected-gradient mapping G_eta(x) = (x - P_K(x - eta grad f(x))) / eta with grad f = 2(x - c)
  (L_phys = 2). At eta = 1/L = 0.5, G = 2 (x - x*), i.e. ||x - x*|| = ||G_{1/L}|| / 2.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

L_PHYS = 2.0
ETA_G = 1.0 / (2.0 * L_PHYS)  # 0.25, step used for the reported stationarity residual ||G||


def project_box(u: np.ndarray, w: float) -> np.ndarray:
    return np.clip(u, -w, w)


def project_ellipsoid(u: np.ndarray, axes: np.ndarray, n_bisect: int = 200) -> np.ndarray:
    """Exact Euclidean projection of rows of ``u`` onto {v : sum (v_i/a_i)^2 <= 1}."""
    u = np.asarray(u, dtype=np.float64)
    a = np.asarray(axes, dtype=np.float64)
    a2 = a ** 2
    shape = u.shape
    u2 = u.reshape(-1, shape[-1])
    inside = np.sum((u2 / a) ** 2, axis=1) <= 1.0
    out = u2.copy()
    idx = np.where(~inside)[0]
    if idx.size:
        uo = u2[idx]
        lo = np.zeros(idx.size)
        hi = np.max(a) * np.linalg.norm(uo, axis=1) + 1.0
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            phi = np.sum((a * uo / (a2 + mid[:, None])) ** 2, axis=1) - 1.0
            pos = phi > 0.0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
        lam = 0.5 * (lo + hi)
        out[idx] = uo * a2 / (a2 + lam[:, None])
    return out.reshape(shape)


class SyntheticGeometry:
    def __init__(self, geometry):
        self.g = geometry
        self.test = geometry.test
        self.B = np.asarray(geometry.basis, dtype=np.float64)  # (D, d)
        self.c = np.asarray(geometry.target, dtype=np.float64)
        if self.test == "test_a":
            self.w = float(geometry.square_half_width)
            self.axes = None
        else:
            self.w = None
            self.axes = np.asarray(geometry.ellipsoid_axes, dtype=np.float64)
        self.x_star = self.project_K(self.c[None])[0]
        self.f_star = float(np.sum((self.x_star - self.c) ** 2))

    def project_C(self, u: np.ndarray) -> np.ndarray:
        if self.test == "test_a":
            return project_box(u, self.w)
        return project_ellipsoid(u, self.axes)

    def project_K(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        u = x @ self.B
        return self.project_C(u) @ self.B.T

    def metrics(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-iterate metrics; ``x`` has shape (..., D) in physical coordinates."""
        x = np.asarray(x, dtype=np.float64)
        shape = x.shape[:-1]
        xf = x.reshape(-1, x.shape[-1])
        u = xf @ self.B
        x_par = u @ self.B.T
        x_perp = xf - x_par
        d_sub = np.linalg.norm(x_perp, axis=1)
        pu = self.project_C(u)
        d_C = np.linalg.norm(u - pu, axis=1)
        dist_K = np.sqrt(d_sub ** 2 + d_C ** 2)
        f = np.sum((xf - self.c) ** 2, axis=1)
        # existing (paper) violation metric, re-implemented verbatim from gss.project_to_feasible
        if self.test == "test_a":
            u_rad = np.clip(u, -self.w, self.w)
        else:
            ns = np.sum((u / self.axes) ** 2, axis=1, keepdims=True)
            u_rad = u * np.where(ns > 1.0, 1.0 / np.sqrt(ns), 1.0)
        viol_existing = np.linalg.norm(xf - u_rad @ self.B.T, axis=1)
        y = xf - ETA_G * 2.0 * (xf - self.c)
        G = (xf - self.project_K(y)) / ETA_G
        out = {
            "objective": f,
            "suboptimality": f - self.f_star,
            "violation_existing": viol_existing,
            "dist_K": dist_K,
            "dist_subspace": d_sub,
            "dist_xstar": np.linalg.norm(xf - self.x_star, axis=1),
            "G_norm": np.linalg.norm(G, axis=1),
        }
        return {k: v.reshape(shape) for k, v in out.items()}


METRIC_KEYS = (
    "objective",
    "suboptimality",
    "violation_existing",
    "dist_K",
    "dist_subspace",
    "dist_xstar",
    "G_norm",
)

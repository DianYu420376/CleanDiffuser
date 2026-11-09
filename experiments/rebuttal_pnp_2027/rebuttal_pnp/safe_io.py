"""No-overwrite, atomic output helpers.

Every artifact is written to a temporary file in the destination directory and
then hard-linked to its final name; ``os.link`` fails if the name exists, so an
existing file is never replaced. Paths must resolve inside ``RESULTS_ROOT`` (or an
explicitly allowed root) and never inside the original repository.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

REBUTTAL_DIR = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = REBUTTAL_DIR.parents[1]
RESULTS_ROOT = REBUTTAL_DIR / "results"
FIGURES_ROOT = REBUTTAL_DIR / "figures"
TABLES_ROOT = REBUTTAL_DIR / "tables"
ORIGINAL_REPO = Path("/u/rzhang26/CleanDiffuser").resolve()
ALLOWED_ROOTS = (RESULTS_ROOT, FIGURES_ROOT, TABLES_ROOT)

# Read-only inputs from the original repository.
ORIG_RESULTS = ORIGINAL_REPO / "results"


class OutputExistsError(FileExistsError):
    pass


def _check(path: Path, roots: Iterable[Path] = ALLOWED_ROOTS) -> Path:
    path = Path(path).resolve()
    if path == ORIGINAL_REPO or ORIGINAL_REPO in path.parents:
        raise PermissionError(f"Refusing to write inside the original repository: {path}")
    if not any(path == r.resolve() or r.resolve() in path.parents for r in roots):
        raise PermissionError(f"Refusing to write outside the rebuttal output roots: {path}")
    return path


def new_run_dir(parent: Path, run_id: Optional[str] = None, tag: str = "") -> Path:
    """Create a fresh run directory; fail if it already exists."""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    run_dir = _check(Path(parent) / run_id)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise OutputExistsError(f"Run directory already exists: {run_dir}") from exc
    return run_dir


def ensure_dir(path: Path) -> Path:
    path = _check(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_bytes(path: Path, data: bytes) -> Path:
    path = _check(path)
    if path.exists():
        raise OutputExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise OutputExistsError(f"Refusing to overwrite existing file: {path}") from exc
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def _to_jsonable(obj: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def write_json(path: Path, obj: Any) -> Path:
    return _atomic_bytes(Path(path), (json.dumps(_to_jsonable(obj), indent=2) + "\n").encode())


def write_text(path: Path, text: str) -> Path:
    return _atomic_bytes(Path(path), text.encode())


def write_npz(path: Path, **arrays) -> Path:
    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return _atomic_bytes(Path(path), buf.getvalue())


def save_figure(fig, path_stem: Path, formats=("png", "pdf"), dpi: int = 300) -> list:
    out = []
    for fmt in formats:
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        out.append(_atomic_bytes(Path(f"{path_stem}.{fmt}"), buf.getvalue()))
    return out


def provenance() -> dict:
    """Git commit / dirty state of the worktree and host info for run metadata."""

    def _git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(WORKTREE_ROOT), *args], text=True).strip()
        except Exception:  # pragma: no cover
            return None

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": (_git("status", "--porcelain") or "").splitlines(),
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

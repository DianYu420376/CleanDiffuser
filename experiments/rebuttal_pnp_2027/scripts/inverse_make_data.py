#!/usr/bin/env python3
"""Generate fresh held-out ground-truth trajectories + observation masks (CPU only).

Uses collect_unicycle_offline_data.collect_dataset with the settings of
run_collect_unicycle_full64.sbatch (only the seed differs). Verifies that no generated
episode coincides with a training episode (initial states) and that val/test are disjoint.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inverse_common as ic  # noqa: E402
from collect_unicycle_offline_data import collect_dataset  # noqa: E402
from rebuttal_pnp import safe_io  # noqa: E402


def _check_sbatch_settings(sbatch_path: Path) -> dict:
    txt = sbatch_path.read_text()
    found = {
        "require_full_horizon": "--require-full-horizon" in txt,
        "horizon": int(re.search(r"--horizon\s+(\d+)", txt).group(1)),
        "target_full_default": int(re.search(r'TARGET_FULL="\$\{TARGET_FULL:-(\d+)\}"', txt).group(1)),
        "seed_default": int(re.search(r'SEED="\$\{SEED:-(\d+)\}"', txt).group(1)),
        "explicit_dt_or_bounds_flags": [f for f in ("--dt", "--x-min", "--x-max", "--y-min", "--y-max", "--v-min",
                                                     "--v-max", "--w-min", "--w-max", "--max-attempts") if f in txt],
    }
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    proto = ic.load_protocol()
    sbatch = ic.WORKTREE / "run_collect_unicycle_full64.sbatch"
    sb = _check_sbatch_settings(sbatch)
    assert sb["require_full_horizon"] and sb["horizon"] == 64 and not sb["explicit_dt_or_bounds_flags"], sb
    # argparse defaults of collect_unicycle_offline_data.main (not overridden by the sbatch)
    settings = dict(horizon=64, dt=0.1, x_lim=(-8.0, 8.0), y_lim=(-8.0, 8.0), v_bounds=(0.0, 2.0),
                    w_bounds=(-1.5, 1.5), require_full_horizon=True, max_attempts=None)
    meta_train = __import__("json").load(open(Path(proto["training_hdf5"]).parent / "metadata.json"))
    for k in ("dt", "horizon"):
        assert meta_train[k] == settings[k], (k, meta_train[k])
    for k in ("x_lim", "y_lim", "v_bounds", "w_bounds"):
        assert tuple(meta_train[k]) == settings[k], (k, meta_train[k])
    assert meta_train["require_full_horizon"] is True

    normalizer, raw_train = ic.training_normalizer(proto["training_hdf5"])
    tr_obs, tr_act = ic.episodes_from_flat(raw_train)
    assert tr_obs.shape == (6000, 64, 4), tr_obs.shape

    # Verify that collect_dataset with these settings and the training seed reproduces the training
    # episodes (the first 50 kept episodes share the RNG stream with the 6000-episode collection).
    rep = collect_dataset(50, settings["horizon"], settings["dt"], int(sb["seed_default"]), settings["x_lim"],
                          settings["y_lim"], settings["v_bounds"], settings["w_bounds"], require_full_horizon=True,
                          target_full_episodes=50, max_attempts=None)
    rep_obs, rep_act = ic.episodes_from_flat(rep)
    train_repro = {"n_checked": 50, "max_abs_obs_diff": float(np.abs(rep_obs - tr_obs[:50]).max()),
                   "max_abs_act_diff": float(np.abs(rep_act - tr_act[:50]).max())}
    print("training-data reproduction check:", train_repro)
    assert train_repro["max_abs_obs_diff"] == 0.0 and train_repro["max_abs_act_diff"] == 0.0, train_repro

    run_dir = safe_io.new_run_dir(safe_io.RESULTS_ROOT / "inverse", run_id=args.run_id, tag="data")
    out = {}
    checks = {}
    for split, seed, n, mseed in (("test", proto["data"]["test_seed"], proto["data"]["n_test"], proto["observations"]["mask_seed_test"]),
                                  ("val", proto["data"]["val_seed"], proto["data"]["n_val"], proto["observations"]["mask_seed_val"])):
        data = collect_dataset(n, settings["horizon"], settings["dt"], seed, settings["x_lim"], settings["y_lim"],
                            settings["v_bounds"], settings["w_bounds"], require_full_horizon=True,
                            target_full_episodes=n, max_attempts=None)
        obs, act = ic.episodes_from_flat(data)
        assert obs.shape == (n, 64, 4) and act.shape == (n, 64, 2)
        # in-distribution / dynamics sanity: next_obs consistent
        d0 = np.linalg.norm(obs[:, None, 0, :] - tr_obs[None, :, 0, :], axis=-1)  # (n, 6000)
        dfull = np.stack([np.abs(obs[j][None] - tr_obs).max(axis=(1, 2)) for j in range(n)])
        obs_idx, M = ic.make_masks(n, mseed, proto["observations"]["n_observed_timesteps"])
        x_star = ic.x_norm_from_phys(normalizer, obs, act)
        Y = M * x_star
        checks[split] = {
            "n": n, "seed": seed, "mask_seed": mseed,
            "min_init_state_dist_to_train": float(d0.min()),
            "min_full_traj_maxabs_dist_to_train": float(dfull.min()),
            "n_init_state_dist_below_1e-6": int((d0 < 1e-6).sum()),
            "xy_range": [float(obs[..., :2].min()), float(obs[..., :2].max())],
            "act_range_v": [float(act[..., 0].min()), float(act[..., 0].max())],
            "act_range_w": [float(act[..., 1].min()), float(act[..., 1].max())],
            "observed_per_traj": int(M[0, :, 0].sum()),
        }
        assert checks[split]["n_init_state_dist_below_1e-6"] == 0
        out[split] = dict(gt_obs=obs, gt_act=act, x_star_norm=x_star, obs_idx=obs_idx, M=M, Y=Y,
                          next_obs=np.stack(np.split(data["next_observations"], n)).astype(np.float32))
    dvt = np.linalg.norm(out["val"]["gt_obs"][:, None, 0] - out["test"]["gt_obs"][None, :, 0], axis=-1)
    checks["min_init_dist_val_vs_test"] = float(dvt.min())
    assert dvt.min() > 1e-6
    for split in out:
        safe_io.write_npz(run_dir / f"{split}_data.npz", **out[split])
    config = {
        "protocol": proto, "protocol_sha256": ic.sha256_file(ic.PROTOCOL_PATH),
        "collect_settings": settings, "collect_sbatch_parsed": sb, "training_metadata": meta_train,
        "training_hdf5_sha256": ic.sha256_file(proto["training_hdf5"]),
        "normalizer": {"source": "UnicycleDataset(load_unicycle_hdf5(training_hdf5), horizon=64).get_normalizer()",
                       "mean": normalizer.mean, "std": normalizer.std},
        "checks": checks, "training_reproduction_check": train_repro, "provenance": safe_io.provenance(),
    }
    safe_io.write_json(run_dir / "data_config.json", config)
    print(__import__("json").dumps(checks, indent=2))
    print("DATA_RUN_DIR", run_dir)


if __name__ == "__main__":
    main()

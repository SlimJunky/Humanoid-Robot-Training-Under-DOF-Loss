from __future__ import annotations

# Copyright (c) 2026, Mikolaj Wyrzykowski
# SPDX-License-Identifier: BSD-3-Clause

'''Behavioral Cloning needs a bc reference demos script which generates many demo episodes for behavioral cloning by randomizing start phase of mapped motion clip. 
Applies random small speed jitters and optionally adding observation / action noise.
Also for Robomimic or Isaac Lab Mimic tools it writes this in a HDf5 style and a JSON summary. Script uses retargeted reference motions as the teacher.
'''
import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# stores environment metadata as JSOn text in HDF5 attributes. Helpers turns python dict into a JSON string.
def as_json_str(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def compute_joint_velocities(joint_targets: np.ndarray, fps: float) -> np.ndarray:
    '''finite-difference joint velocities for [T, J] joint targets.'''
    if joint_targets.ndim != 2:
        raise ValueError(f"Expected joint_targets shape [T, J], got {joint_targets.shape}")
    dt = 1.0 / fps
    qd = np.zeros_like(joint_targets, dtype=np.float32)
    if joint_targets.shape[0] == 1:
        return qd
    # forward/backward at edges, central difference in middle
    qd[0] = (joint_targets[1] - joint_targets[0]) / dt
    qd[-1] = (joint_targets[-1] - joint_targets[-2]) / dt
    if joint_targets.shape[0] > 2:
        qd[1:-1] = (joint_targets[2:] - joint_targets[:-2]) / (2.0 * dt)
    return qd.astype(np.float32)


# from numpy arrays in .npz first pass that is mapped. observation vector for BC.
def build_obs(
    q: np.ndarray,
    qd: np.ndarray,
    phase: np.ndarray,
    include_phase: bool,
) -> np.ndarray:
    """Build low-dimensional proprioceptive observation."""
    pieces = [q.astype(np.float32), qd.astype(np.float32)]
    if include_phase:
        pieces.append(phase[:, None].astype(np.float32))
    return np.concatenate(pieces, axis=1)

def normalize_actions(actions: np.ndarray, action_lo: np.ndarray, action_hi: np.ndarray) -> np.ndarray:
    '''Map raw joint targets to [-1, 1] using per-dimension limits. This is required for robomimic so made this function and its denormalizing'''
    denom = action_hi - action_lo
    #avoid divide-by-zero if any range is degenerate
    denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
    normed = 2.0 * (actions - action_lo[None, :]) / denom[None, :] - 1.0
    return np.clip(normed, -1.0, 1.0).astype(np.float32)

def denormalize_actions(actions_norm: np.ndarray, action_lo: np.ndarray, action_hi: np.ndarray) -> np.ndarray:
    '''map normalized [-1, 1] actions back to raw joint targets'''
    return (((actions_norm + 1.0) * 0.5) * (action_hi - action_lo)[None, :] + action_lo[None, :]).astype(np.float32)

'''what generates my one demonstration episode from mapped reference motion. This forces BC teacher as taken from Isaac Sim docs to be:
given the current robot state, predict the next desired joint target. robomimic expects trajectory-like fields in dataset
rwards are one. dones are set to True and states as dummy zero array.'''

def make_rollout(
    joint_targets: np.ndarray,
    fps: float,
    action_lo: np.ndarray,
    action_hi: np.ndarray,
    start_idx: int = 0,
    speed_scale: float = 1.0,
    obs_noise_std: float = 0.0,
    action_noise_std: float = 0.0,
    include_phase: bool = True,
    normalize_action_targets: bool = True,
) -> dict[str, np.ndarray]:
    T, J = joint_targets.shape
    if T < 2:
        raise ValueError("Need at least 2 frames to generate a demo rollout.")

    # Circularly shift starting phase This means each demo can start at different point in walk cycle for better reference motion training
    q_seq = np.roll(joint_targets, -start_idx, axis=0).astype(np.float32)

    #ptional playback speed scaling by resampling in normalized time
    if not np.isclose(speed_scale, 1.0):
        src_t = np.arange(T, dtype=np.float64)
        dst_t = np.arange(T, dtype=np.float64) * speed_scale
        dst_t = np.clip(dst_t, 0.0, T - 1)
        q_resampled = np.zeros_like(q_seq, dtype=np.float32)
        for j in range(J):
            q_resampled[:, j] = np.interp(dst_t, src_t, q_seq[:, j]).astype(np.float32)
        q_seq = q_resampled

    qd_seq = compute_joint_velocities(q_seq, fps)

    phase = np.linspace(0.0, 1.0, T, endpoint=False, dtype=np.float32)

    # State at time t predicts action for time t+1
    obs = build_obs(q_seq[:-1], qd_seq[:-1], phase[:-1], include_phase)
    next_obs = build_obs(q_seq[1:], qd_seq[1:], phase[1:], include_phase)
    raw_actions = q_seq[1:].copy().astype(np.float32)
    if normalize_action_targets:
        actions = normalize_actions(raw_actions, action_lo, action_hi)
    else:
        actions = raw_actions

    if obs_noise_std > 0.0:
        obs += np.random.normal(0.0, obs_noise_std, size=obs.shape).astype(np.float32)
        next_obs += np.random.normal(0.0, obs_noise_std, size=next_obs.shape).astype(np.float32)

    if action_noise_std > 0.0:
        actions += np.random.normal(0.0, action_noise_std, size=actions.shape).astype(np.float32)
        if normalize_action_targets:
            actions = np.clip(actions, -1.0, 1.0).astype(np.float32)

    rewards = np.ones((actions.shape[0],), dtype=np.float32)
    dones = np.zeros((actions.shape[0],), dtype=np.bool_)
    dones[-1] = True
    states = np.zeros((actions.shape[0], 1), dtype=np.float32)

    return {
        "obs": obs,
        "next_obs": next_obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "states": states,
    }

# Creates the data ready for robomimic BC training

def write_robomimic_hdf5(
    output_hdf5: Path,
    demos: list[dict[str, np.ndarray]],
    env_args: dict[str, Any],
    action_keys: list[str],
    obs_keys: list[str],
) -> None:
    # Write a robomimic-style HDF5 dataset.
    ensure_dir(output_hdf5.parent)
    with h5py.File(output_hdf5, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["env_args"] = as_json_str(env_args)
        total = 0

        for i, demo in enumerate(demos):
            demo_grp = data_grp.create_group(f"demo_{i}")
            demo_grp.attrs["num_samples"] = int(demo["actions"].shape[0])

            demo_grp.create_dataset("actions", data=demo["actions"], compression="gzip")
            demo_grp.create_dataset("rewards", data=demo["rewards"], compression="gzip")
            demo_grp.create_dataset("dones", data=demo["dones"], compression="gzip")
            demo_grp.create_dataset("states", data=demo["states"], compression="gzip")

            obs_grp = demo_grp.create_group("obs")
            next_obs_grp = demo_grp.create_group("next_obs")

            obs_key = obs_keys[0]
            obs_grp.create_dataset(obs_key, data=demo["obs"], compression="gzip")
            next_obs_grp.create_dataset(obs_key, data=demo["next_obs"], compression="gzip")

            total += int(demo["actions"].shape[0])

        data_grp.attrs["total"] = total
        data_grp.attrs["action_keys"] = json.dumps(action_keys)
        data_grp.attrs["obs_keys"] = json.dumps(obs_keys)


def main():
    # Main arguments when generating this
    parser = argparse.ArgumentParser(
        description="Create robomimic-style HDF5 BC demos from a mapped G1 reference motion."
    )
    parser.add_argument("mapped_npz", help="Path to mapped G1 motion npz (from g1_map_* script).")
    parser.add_argument(
        "--out-hdf5",
        default="datasets/g1_walk_reference_bc.hdf5",
        help="Output HDF5 path for robomimic-style demos.",
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=256,
        help="Number of reference-driven demo episodes to generate.",
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=0.005,
        help="Gaussian noise std added to observations.",
    )
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std added to target actions.",
    )
    parser.add_argument(
        "--speed-jitter",
        type=float,
        default=0.05,
        help="Uniform playback-speed jitter fraction, e.g. 0.05 => sample in [0.95, 1.05].",
    )
    #NORMALZIED MOTION PHASE IMPORTANT. ROBOMIMIC EXPECTS NORMALIZED MOTION START ASWELL
    parser.add_argument(
        "--include-phase",
        action="store_true",
        help="Include normalized motion phase in low-dim observations.",
    )
    parser.add_argument(
        "--no-normalize-actions",
        action="store_true",
        help="Store raw joint-target actions instead of normalized [-1, 1] actions.",
    )
    # Will generate tons of reference motions at "random" based on numpy random seed function
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    mapped_path = Path(args.mapped_npz)
    out_hdf5 = Path(args.out_hdf5)

    if not mapped_path.exists():
        raise FileNotFoundError(f"Mapped motion not found: {mapped_path}")

    with np.load(mapped_path, allow_pickle=True) as data:
        joint_targets = np.asarray(data["joint_targets"], dtype=np.float32)
        joint_names = [str(x) for x in data["joint_names"].tolist()]
        category = str(np.asarray(data["category"]).item()) if "category" in data else "unknown"

        if "soft_joint_lower" not in data or "soft_joint_upper" not in data:
            raise KeyError(
                "Mapped npz must contain 'soft_joint_lower' and 'soft_joint_upper' "
                "for action normalization."
            )

        action_lo = np.asarray(data["soft_joint_lower"], dtype=np.float32)
        action_hi = np.asarray(data["soft_joint_upper"], dtype=np.float32)

        if "source_fps" in data:
            fps = float(np.asarray(data["source_fps"]).item())
        elif "time_s" in data and len(data["time_s"]) > 1:
            time_s = np.asarray(data["time_s"], dtype=np.float64)
            fps = 1.0 / float(time_s[1] - time_s[0])
        else:
            fps = 60.0

    demos: list[dict[str, np.ndarray]] = []
    T = joint_targets.shape[0]

    for i in range(args.num_demos):
        start_idx = int(np.random.randint(0, T))
        speed_scale = float(np.random.uniform(1.0 - args.speed_jitter, 1.0 + args.speed_jitter))
        demos.append(
            make_rollout(
                joint_targets=joint_targets,
                fps=fps,
                action_lo=action_lo,
                action_hi=action_hi,
                start_idx=start_idx,
                speed_scale=speed_scale,
                obs_noise_std=args.obs_noise_std,
                action_noise_std=args.action_noise_std,
                include_phase=args.include_phase,
                normalize_action_targets=not args.no_normalize_actions,
            )
        )

    obs_dim = demos[0]["obs"].shape[1]
    act_dim = demos[0]["actions"].shape[1]

    '''metadata stored in HDF5 so training & evaluation code knows what environment dataset belongs to. Robomimics environment metadata requirements'''
    env_args = {
        "env_name": "IsaacLabG1ReferenceBC", #Environment ID
        "env_type": 2,
        "type": 2, #correct environment wrapper class for custom gym-style environment
        "env_version": "1.0",
        "env_kwargs": {
            "robot": "G1_MINIMAL_CFG",
            "category": category,
            "obs_key": "proprio", #name of observation entry inside. low-dimension proprioception of joint positions and joint velocities.
            "obs_dim": obs_dim, # size of observation vector
            "act_dim": act_dim, # size of action vector
            "joint_names": joint_names,
            "teacher": "retargeted_reference_motion",
            "source_mapped_npz": str(mapped_path.resolve()),
            "actions_normalized": not args.no_normalize_actions,
            "action_normalization": {
                "method": "per_joint_soft_limit_affine_map" if not args.no_normalize_actions else "none",
                "low": action_lo.tolist(),
                "high": action_hi.tolist(),
            },
            "notes": (
                "Reference-driven BC dataset generated from mapped G1 motion. "
                "This is suitable as a custom robomimic-style low-dimensional dataset, "
                "but a custom environment wrapper may still be needed during training."
            ),
        },
    }

    write_robomimic_hdf5(
        output_hdf5=out_hdf5,
        demos=demos,
        env_args=env_args,
        action_keys=["actions"],
        obs_keys=["proprio"],
    )

    meta_path = out_hdf5.with_suffix(".json")
    meta = {
        "output_hdf5": str(out_hdf5.resolve()),
        "num_demos": args.num_demos,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "fps": fps,
        "joint_names": joint_names,
        "source_mapped_npz": str(mapped_path.resolve()),
        "include_phase": args.include_phase,
        "obs_noise_std": args.obs_noise_std,
        "action_noise_std": args.action_noise_std,
        "speed_jitter": args.speed_jitter,
        "actions_normalized": not args.no_normalize_actions,
        "action_low": action_lo.tolist(),
        "action_high": action_hi.tolist(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved HDF5 dataset : {out_hdf5}")
    print(f"Saved metadata JSON : {meta_path}")
    print(f"Num demos          : {args.num_demos}")
    print(f"Obs dim            : {obs_dim}")
    print(f"Action dim         : {act_dim}")
    print(f"Actions normalized : {not args.no_normalize_actions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

# Copyright (c) 2026, Mikolaj Wyrzykowski
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import robomimic.utils.file_utils as FileUtils


def build_qd(reference_q: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / fps
    qd = np.zeros_like(reference_q, dtype=np.float32)

    qd[1:-1] = (reference_q[2:] - reference_q[:-2]) / (2.0 * dt)
    qd[0] = (reference_q[1] - reference_q[0]) / dt
    qd[-1] = (reference_q[-1] - reference_q[-2]) / dt

    return qd.astype(np.float32)


def build_obs(q: np.ndarray, qd: np.ndarray, phase: float, include_phase: bool) -> np.ndarray:
    parts = [q.astype(np.float32), qd.astype(np.float32)]

    if include_phase:
        parts.append(np.array([phase], dtype=np.float32))

    return np.concatenate(parts, axis=0).astype(np.float32)


def denormalize(action_norm: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    action_norm = np.clip(action_norm, -1.0, 1.0)
    return 0.5 * (action_norm + 1.0) * (action_high - action_low) + action_low

'''Call the original robomimic .pth policy network in the same way the TorchScript exporter does.'''
def call_pth_policy(rollout_policy, obs_vec: np.ndarray, device: str) -> np.ndarray:
    algo = rollout_policy.policy

    if hasattr(algo, "set_eval"):
        algo.set_eval()

    if hasattr(algo, "reset"):
        algo.reset()

    policy_net = algo.nets["policy"]
    policy_net.eval()

    obs_tensor = torch.tensor(obs_vec[None, :], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = policy_net(obs_dict={"proprio": obs_tensor}, goal_dict=None)

    if isinstance(out, (tuple, list)):
        out = out[0]

    return out.detach().cpu().numpy()[0].astype(np.float32)

"""Call the exported TorchScript .pt policy with the same plain obs_vec input."""
def call_torchscript_policy(ts_policy, obs_vec: np.ndarray, device: str) -> np.ndarray:
    obs_tensor = torch.tensor(obs_vec[None, :], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = ts_policy(obs_tensor)

    if isinstance(out, (tuple, list)):
        out = out[0]

    if not isinstance(out, torch.Tensor):
        out = torch.as_tensor(out, device=device)

    return out.detach().cpu().numpy()[0].astype(np.float32)


def print_joint_sample(label: str, values: np.ndarray, joint_names: list[str]) -> None:
    sample_names = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "torso_joint",
    ]

    parts = []
    for name in sample_names:
        if name in joint_names:
            idx = joint_names.index(name)
            parts.append(f"{name}={values[idx]:+.4f}")

    print(f"    {label}: " + ", ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare robomimic .pth BC policy against exported TorchScript .pt policy.")
    parser.add_argument("--pth", required=True, help="Path to robomimic .pth checkpoint")
    parser.add_argument("--pt", required=True, help="Path to exported TorchScript .pt policy")
    parser.add_argument("--meta", required=True, help="Path to BC metadata JSON")
    parser.add_argument("--source-npz", type=str, default=None, help="Optional override path for source_mapped_npz, useful when metadata contains a Windows path.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--num-random", type=int, default=5)
    args = parser.parse_args()

    pth_path = Path(args.pth)
    pt_path = Path(args.pt)
    meta_path = Path(args.meta)

    if not pth_path.exists():
        raise FileNotFoundError(f"Missing .pth checkpoint: {pth_path}")
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing .pt policy: {pt_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata JSON: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    joint_names = meta["joint_names"]
    action_low = np.asarray(meta["action_low"], dtype=np.float32)
    action_high = np.asarray(meta["action_high"], dtype=np.float32)

    obs_dim = int(meta.get("obs_dim", 75))
    act_dim = int(meta.get("act_dim", len(joint_names)))
    fps = float(meta.get("fps", 60.0))
    include_phase = bool(meta.get("include_phase", obs_dim == 75))

    print("[INFO] Loading robomimic .pth policy...")
    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(pth_path),
        device=args.device,
    )

    print("[INFO] Loading TorchScript .pt policy...")
    ts_policy = torch.jit.load(str(pt_path), map_location=args.device)
    ts_policy.eval()

    print(f"[INFO] obs_dim={obs_dim}, act_dim={act_dim}, include_phase={include_phase}, fps={fps}")
    print(f"[INFO] joint_count={len(joint_names)}")

    source_npz_raw = args.source_npz if args.source_npz is not None else meta["source_mapped_npz"]
    source_npz = Path(source_npz_raw)

    if not source_npz.exists():
        # Helpful fallback: if metadata has a Windows path, try finding the same filename beside the JSON.
        fallback_npz = meta_path.parent / Path(str(meta["source_mapped_npz"]).replace("\\", "/")).name

        if fallback_npz.exists():
            source_npz = fallback_npz
            print(f"[INFO] Using fallback source_mapped_npz beside metadata: {source_npz}")
        else:
            raise FileNotFoundError(
                "source_mapped_npz does not exist.\n"
                f"Requested path: {source_npz}\n"
                f"Fallback tried: {fallback_npz}\n"
                "Pass the Linux path manually with --source-npz."
            )

    print(f"[INFO] source_mapped_npz: {source_npz}")

    with np.load(source_npz, allow_pickle=True) as data:
        reference_q = np.asarray(data["joint_targets"], dtype=np.float32)

    reference_qd = build_qd(reference_q, fps)
    total_frames = reference_q.shape[0]

    print(f"[INFO] reference_q shape={reference_q.shape}")

    # Test useful real rollout frames across the gait cycle.
    frame_ids = sorted(set([
        0,
        total_frames // 8,
        total_frames // 4,
        3 * total_frames // 8,
        total_frames // 2,
        5 * total_frames // 8,
        3 * total_frames // 4,
        7 * total_frames // 8,
        total_frames - 2,
    ]))

    # In addition test some noisy versions around reference frames
    rng = np.random.default_rng(0)

    max_raw_diffs = []
    max_q_diffs = []
    mean_qref_errors_pt = []
    mean_qref_errors_pth = []

    print("\n[COMPARE] Testing reference-frame observations\n")

    for i, frame_idx in enumerate(frame_ids):
        phase = float(frame_idx / total_frames)

        obs = build_obs(
            reference_q[frame_idx],
            reference_qd[frame_idx],
            phase,
            include_phase,
        )

        if obs.shape[0] != obs_dim:
            raise RuntimeError(f"Observation dimension mismatch: built {obs.shape[0]}, expected {obs_dim}")

        action_pth = call_pth_policy(rollout_policy, obs, args.device)
        action_pt = call_torchscript_policy(ts_policy, obs, args.device)

        if action_pth.shape[0] != act_dim:
            raise RuntimeError(f".pth action dim mismatch: got {action_pth.shape[0]}, expected {act_dim}")
        if action_pt.shape[0] != act_dim:
            raise RuntimeError(f".pt action dim mismatch: got {action_pt.shape[0]}, expected {act_dim}")

        raw_diff = np.abs(action_pth - action_pt)

        q_pth = denormalize(action_pth, action_low, action_high)
        q_pt = denormalize(action_pt, action_low, action_high)

        q_diff = np.abs(q_pth - q_pt)

        next_idx = (frame_idx + 1) % total_frames
        q_ref_next = reference_q[next_idx]

        qref_error_pth = np.abs(q_pth - q_ref_next)
        qref_error_pt = np.abs(q_pt - q_ref_next)

        max_raw_diffs.append(float(raw_diff.max()))
        max_q_diffs.append(float(q_diff.max()))
        mean_qref_errors_pth.append(float(qref_error_pth.mean()))
        mean_qref_errors_pt.append(float(qref_error_pt.mean()))

        print(
            f"[FRAME {frame_idx:04d} phase={phase:.3f}] "
            f"raw_action_diff max={raw_diff.max():.8f}, mean={raw_diff.mean():.8f} | "
            f"q_target_diff max={q_diff.max():.8f}, mean={q_diff.mean():.8f} | "
            f"mean|pth_q-ref_next|={qref_error_pth.mean():.5f} | "
            f"mean|pt_q-ref_next|={qref_error_pt.mean():.5f}"
        )

        print_joint_sample("pth q_target", q_pth, joint_names)
        print_joint_sample("pt  q_target", q_pt, joint_names)
        print_joint_sample("ref next", q_ref_next, joint_names)
        print()

    print("\n[COMPARE] Testing random noisy observations around reference frames\n")

    for i in range(args.num_random):
        frame_idx = int(rng.integers(0, total_frames))
        phase = float(frame_idx / total_frames)

        q = reference_q[frame_idx] + rng.normal(0.0, 0.02, size=act_dim).astype(np.float32)
        qd = reference_qd[frame_idx] + rng.normal(0.0, 0.05, size=act_dim).astype(np.float32)

        obs = build_obs(q, qd, phase, include_phase)

        action_pth = call_pth_policy(rollout_policy, obs, args.device)
        action_pt = call_torchscript_policy(ts_policy, obs, args.device)

        raw_diff = np.abs(action_pth - action_pt)

        print(
            f"[RANDOM {i} frame={frame_idx:04d}] "
            f"raw_action_diff max={raw_diff.max():.8f}, mean={raw_diff.mean():.8f}"
        )

        max_raw_diffs.append(float(raw_diff.max()))

    print("\n================ SUMMARY ================")
    print(f"Max raw action diff across all tests : {max(max_raw_diffs):.10f}")
    print(f"Max q_target diff on reference frames: {max(max_q_diffs):.10f}")
    print(f"Mean .pth q/ref_next error          : {np.mean(mean_qref_errors_pth):.6f}")
    print(f"Mean .pt  q/ref_next error          : {np.mean(mean_qref_errors_pt):.6f}")

    if max(max_raw_diffs) < 1e-4:
        print("[OK] Exported .pt matches robomimic .pth very closely.")
        print("[OK] The weak walking is probably from the reference/BC policy quality, not the export.")
    elif max(max_raw_diffs) < 1e-2:
        print("[WARN] .pt is close but not identical to .pth. Probably acceptable, but inspect normalization/settings.")
    else:
        print("[BAD] .pt does not match .pth. The export wrapper is likely bypassing something important.")

    print("=========================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
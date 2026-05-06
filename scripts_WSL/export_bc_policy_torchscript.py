from __future__ import annotations

# Copyright (c) 2026, Mikolaj Wyrzykowski
# SPDX-License-Identifier: BSD-3-Clause

"""Export a robomimic BC .pth checkpoint into a plain TorchScript .pt policy. Input: obs_vec: [B, 75]  to [q, qd, phase]
Output: action: [B, 37] into normalized action. All BC trained on normalized actions. 
This exported .pt can then be loaded from the Windows Isaac Lab G1 playback script without needing robomimic installed in the Windows Isaac Lab environment.
"""

'''Warning, this will not work unless you are in described robomimic environment'''

import argparse
from pathlib import Path

import torch
import robomimic.utils.file_utils as FileUtils


class ProprioPolicyWrapper(torch.nn.Module):
    """ Wrap the underlying robomimic torch policy network so it accepts a plain tensor: obs_vec: [B, 75]
    and internally converts it into the robomimic observation dict: {"proprio": obs_vec} returning: action: [B, 37]
    """

    def __init__(self, algo):
        super().__init__()

        # algo is the underlying robomimic Algo object.
        # algo.nets["policy"] is the actual torch policy network.
        self.policy_net = algo.nets["policy"]

    def forward(self, obs_vec: torch.Tensor) -> torch.Tensor:
        obs = {"proprio": obs_vec}

        # robomimic ActorNetwork expects obs_dict and optional goal_dict.
        out = self.policy_net(obs_dict=obs, goal_dict=None)

        # Different return policy styles
        if isinstance(out, (tuple, list)):
            out = out[0]

        return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export robomimic BC checkpoint to TorchScript .pt")
    parser.add_argument("ckpt", type=str, help="Path to robomimic checkpoint .pth")
    parser.add_argument("out_pt", type=str, help="Output TorchScript .pt path")
    parser.add_argument("--device", type=str, default="cuda", help="Device for loading and tracing")
    parser.add_argument("--obs-dim", type=int, default=75, help="Observation dimension")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out_pt)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading robomimic checkpoint: {ckpt_path}")

    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(ckpt_path),
        device=args.device,
    )

    print(f"[INFO] Loaded object type: {type(rollout_policy)}")

    # policy_from_checkpoint returns a RolloutPolicy.
    # The actual robomimic object is stored inside rollout_policy.policy. Took me ages to figure out
    algo = rollout_policy.policy

    print(f"[INFO] Underlying algo type: {type(algo)}")

    # Robomimic Algo objects use set_eval(), not eval().
    algo.set_eval()
    algo.reset()

    wrapped = ProprioPolicyWrapper(algo).to(args.device).eval()

    example = torch.zeros(1, args.obs_dim, dtype=torch.float32, device=args.device)

    print(f"[INFO] Testing wrapped policy with input shape: {tuple(example.shape)}")

    with torch.no_grad():
        test_out = wrapped(example)

    if not isinstance(test_out, torch.Tensor):
        raise TypeError(f"Wrapped policy output is not a Tensor. Got: {type(test_out)}")

    print(f"[INFO] Wrapped policy output shape: {tuple(test_out.shape)}")
    print(
        f"[INFO] Output stats: "
        f"min={test_out.min().item():+.6f}, "
        f"max={test_out.max().item():+.6f}, "
        f"mean={test_out.mean().item():+.6f}"
    )

    if torch.isnan(test_out).any():
        raise RuntimeError("Wrapped policy produced NaN before tracing.")

    print(f"[INFO] Tracing policy with example input shape: {tuple(example.shape)}")

    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example, strict=False)

    # Test traced policy before saving
    with torch.no_grad():
        traced_out = traced(example)

    print(f"[INFO] Traced policy output shape: {tuple(traced_out.shape)}")

    if torch.isnan(traced_out).any():
        raise RuntimeError("Traced policy produced NaN.")

    traced.save(str(out_path))

    print(f"[INFO] Saved TorchScript policy to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
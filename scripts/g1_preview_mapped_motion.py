from __future__ import annotations

# Copyright (c) 2026, Mikolaj Wyrzykowski
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Preview a mapped G1 motion clip by directly writing joint states."
)
parser.add_argument("mapped_npz", type=str, help="Path to mapped G1 motion npz.")
parser.add_argument(
    "--fps",
    type=float,
    default=None,
    help="Override playback FPS. If omitted, uses time_s from file or defaults to 60.",
)
parser.add_argument("--loop", action="store_true", help="Loop the motion continuously.")
parser.add_argument(
    "--root-height",
    type=float,
    default=0.78,
    help="Fixed root Z height for visual preview.",
)
# Just in-case in preview it does not work properly.
parser.add_argument(
    "--root-x",
    type=float,
    default=0.0,
    help="Fixed root X position for preview.",
)
parser.add_argument(
    "--root-y",
    type=float,
    default=0.0,
    help="Fixed root Y position for preview.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets import G1_MINIMAL_CFG


def main():
    mapped_path = Path(args_cli.mapped_npz)
    if not mapped_path.exists():
        raise FileNotFoundError(f"Mapped motion not found: {mapped_path}")

    with np.load(mapped_path, allow_pickle=True) as data:
        joint_targets_np = np.asarray(data["joint_targets"], dtype=np.float64)
        joint_names_file = [str(x) for x in data["joint_names"].tolist()]
        time_s_np = np.asarray(data["time_s"], dtype=np.float64) if "time_s" in data else None
        default_joint_pos_np = np.asarray(data["default_joint_pos"], dtype=np.float64)

    num_frames, num_joints = joint_targets_np.shape

    #Simulation CFG
    sim_cfg = SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.8, 2.0, 1.8], [0.0, 0.0, 0.8])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sim_utils.create_prim("/World/Origin1", "Xform", translation=[0.0, 0.0, 0.0])

    #Robot CFG
    g1_cfg = G1_MINIMAL_CFG.copy()
    g1_cfg.prim_path = "/World/Origin1/Robot"
    robot = Articulation(cfg=g1_cfg)

    sim.reset()

    # Verify joint order matches the mapped file
    robot_joint_names = list(robot.data.joint_names)
    if robot_joint_names != joint_names_file:
        print("[ERROR] Joint order mismatch between mapped file and current G1 articulation.")
        print("[ERROR] File joints : ", joint_names_file)
        print("[ERROR] Robot joints: ", robot_joint_names)
        raise RuntimeError("Joint name/order mismatch. Rebuild the mapped file against this asset.")

    # Put robot in a visible fixed-root pose
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0] = args_cli.root_x
    root_state[:, 1] = args_cli.root_y
    root_state[:, 2] = args_cli.root_height

    default_joint_pos = torch.tensor(default_joint_pos_np, dtype=torch.float32, device=robot.device).unsqueeze(0)
    zero_joint_vel = torch.zeros_like(default_joint_pos)

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(default_joint_pos, zero_joint_vel)
    robot.reset()
    robot.update(sim.get_physics_dt())

    # Playback Timing
    if args_cli.fps is not None:
        playback_fps = float(args_cli.fps)
    elif time_s_np is not None and len(time_s_np) > 1:
        dt_motion = float(time_s_np[1] - time_s_np[0])
        playback_fps = 1.0 / dt_motion
    else:
        playback_fps = 60.0

    sim_dt = sim.get_physics_dt()
    sim_steps_per_frame = max(1, int(round((1.0 / playback_fps) / sim_dt)))

    print(f"[INFO] Loaded mapped motion: {mapped_path}")
    print(f"[INFO] Frames: {num_frames}")
    print(f"[INFO] Joints: {num_joints}")
    print(f"[INFO] Playback FPS: {playback_fps:.3f}")
    print(f"[INFO] Sim dt: {sim_dt:.6f}")
    print(f"[INFO] Sim steps per motion frame: {sim_steps_per_frame}")

    frame_idx = 0
    step_count = 0

    while simulation_app.is_running():
        if step_count % sim_steps_per_frame == 0:
            q = torch.tensor(joint_targets_np[frame_idx], dtype=torch.float32, device=robot.device).unsqueeze(0)

            # Keep the base/root fixed for a clean kinematic preview
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            # Directly write the joint state for visual inspection
            robot.write_joint_state_to_sim(q, zero_joint_vel)
            robot.reset()

            frame_idx += 1
            if frame_idx >= num_frames:
                if args_cli.loop:
                    frame_idx = 0
                else:
                    frame_idx = num_frames - 1

        sim.step()
        robot.update(sim_dt)
        step_count += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

# Copyright (c) 2026, Mikolaj Wyrzykowski
# SPDX-License-Identifier: BSD-3-Clause

''' Empty world in isaac lab only for Unitree G1 asset taken from similar environments in tutorial NVIDIA Isaac Lab. 
script to gather information about G1 asset in selected empty world for retargeting.'''

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "source").exists():
            return parent
    raise RuntimeError("Could not find project root. Expected pyproject.toml and source/ folder.")


PROJECT_ROOT = find_project_root()


#  Parser arguments for script running and output save location
parser = argparse.ArgumentParser(description="Spawn Unitree G1 in a flat world and dump articulation info.")
parser.add_argument("--out", type=str, default="outputs/g1_asset_dump.json", help="Path to output JSON dump.")
parser.add_argument("--stay-open", action="store_true", help="Keep Isaac Sim open after dumping info.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Default IsaacLab Imports that require the app to be running
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext

# Using the official minimal G1 config used by Isaac Lab locomotion. G1 Minimal 23 DOF usage. Faster in simulation and for RL training.
from isaaclab_assets import G1_MINIMAL_CFG


def to_serializable(x):
    """Convert tensors / numpy / misc objects into JSON-safe Python values."""
    # torch tensors
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().tolist()
    # numpy arrays / scalars
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    # pathlib
    if isinstance(x, Path):
        return str(x)
    # lists / tuples
    if isinstance(x, (list, tuple)):
        return [to_serializable(v) for v in x]
    # dicts
    if isinstance(x, dict):
        return {str(k): to_serializable(v) for k, v in x.items()}
    # primitives
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def main():

    # Simulation
    sim_cfg = SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 1.8], [0.0, 0.0, 0.8])

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Spawn transform
    sim_utils.create_prim("/World/Origin1", "Xform", translation=[0.0, 0.0, 0.0])

   
    # Robot
    g1_cfg = G1_MINIMAL_CFG.copy()
    g1_cfg.prim_path = "/World/Origin1/Robot"
    robot = Articulation(cfg=g1_cfg)

    sim.reset()

    # Reset robot into default state once
    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    robot.update(sim.get_physics_dt())


    #Extract asset information after spawning from Origin 1 Robot and get all the joint names and initial positions.
    dump = {
        "robot_cfg": "G1_MINIMAL_CFG",
        "prim_path": g1_cfg.prim_path,
        "num_joints": len(robot.data.joint_names),
        "num_bodies": len(robot.data.body_names),
        "joint_names": to_serializable(robot.data.joint_names),
        "body_names": to_serializable(robot.data.body_names),
        "default_root_state": to_serializable(robot.data.default_root_state),
        "default_joint_pos": to_serializable(robot.data.default_joint_pos),
        "default_joint_vel": to_serializable(robot.data.default_joint_vel),
        "joint_pos_limits": to_serializable(robot.data.joint_pos_limits),
        "soft_joint_pos_limits": to_serializable(robot.data.soft_joint_pos_limits),
    }

    # Helpful flattened joint table for quick inspection
    joint_table = []
    joint_names = robot.data.joint_names
    default_pos = robot.data.default_joint_pos[0].detach().cpu().tolist()
    default_vel = robot.data.default_joint_vel[0].detach().cpu().tolist()
    joint_limits = robot.data.joint_pos_limits[0].detach().cpu().tolist()
    soft_limits = robot.data.soft_joint_pos_limits[0].detach().cpu().tolist()

    for i, name in enumerate(joint_names):
        joint_table.append(
            {
                "index": i,
                "name": name,
                "default_pos": default_pos[i],
                "default_vel": default_vel[i],
                "limit_lower": joint_limits[i][0],
                "limit_upper": joint_limits[i][1],
                "soft_limit_lower": soft_limits[i][0],
                "soft_limit_upper": soft_limits[i][1],
            }
        )

    dump["joint_table"] = joint_table

    #JSON
    out_path = Path(args_cli.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_serializable(dump), indent=2), encoding="utf-8")


    print(f"[INFO] Saved G1 asset dump to: {out_path}")
    print(f"[INFO] Num joints: {dump['num_joints']}")
    print(f"[INFO] Num bodies: {dump['num_bodies']}")
    print("[INFO] Joint names in articulation order:")
    for item in joint_table:
        print(f"  {item['index']:02d}: {item['name']} | default={item['default_pos']:.6f} | "
              f"limits=({item['limit_lower']:.6f}, {item['limit_upper']:.6f})")

    if args_cli.stay_open:
        print("[INFO] Scene is open. Close the app window or Ctrl+C to stop.")
        while simulation_app.is_running():
            sim.step()
            robot.update(sim.get_physics_dt())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
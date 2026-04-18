from __future__ import annotations

import argparse
import traceback # Debug check for sim startup
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

# Parser arguments
parser = argparse.ArgumentParser(
    description="Physics-based playback of mapped G1 joint targets."
)
parser.add_argument("mapped_npz", type=str, help="Path to mapped G1 motion npz.")
parser.add_argument(
    "--fps",
    type=float,
    default=None,
    help="Override motion playback FPS. If omitted, use time_s from file or default to 60.",
)
parser.add_argument("--loop", action="store_true", help="Loop the motion.")
parser.add_argument(
    "--root-height",
    type=float,
    default=0.78,
    help="Initial root/base Z height when resetting the robot.",
)
parser.add_argument(
    "--root-x",
    type=float,
    default=0.0,
    help="Initial root/base X position when resetting the robot.",
)
parser.add_argument(
    "--root-y",
    type=float,
    default=0.0,
    help="Initial root/base Y position when resetting the robot.",
)
parser.add_argument(
    "--reset-on-loop",
    action="store_true",
    help="Reset robot pose/state when the motion loops.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets import G1_MINIMAL_CFG


def reset_robot(robot: Articulation, default_joint_pos_np: np.ndarray, root_x: float, root_y: float, root_z: float) -> None:
    """Reset root and joint state into a clean starting pose."""
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0] = root_x
    root_state[:, 1] = root_y
    root_state[:, 2] = root_z

    joint_pos = torch.tensor(default_joint_pos_np, dtype=torch.float32, device=robot.device).unsqueeze(0)
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()


def main():
    mapped_path = Path(args_cli.mapped_npz)
    if not mapped_path.exists():
        raise FileNotFoundError(f"Mapped motion not found: {mapped_path}")

    with np.load(mapped_path, allow_pickle=True) as data:
        joint_targets_np = np.asarray(data["joint_targets"], dtype=np.float64)
        joint_names_file = [str(x) for x in data["joint_names"].tolist()]
        time_s_np = np.asarray(data["time_s"], dtype=np.float64) if "time_s" in data else None
        default_joint_pos_np = np.asarray(data["default_joint_pos"], dtype=np.float64)
    
    print("[CHK] mapped path exists")


    num_frames, num_joints = joint_targets_np.shape

    #SIMULATION CONFIG
    sim_cfg = SimulationCfg(dt=1.0 / 120.0, device=args_cli.device) # Need to match conda environment expected device.
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.8, 2.0, 1.8], [0.0, 0.0, 0.8])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sim_utils.create_prim("/World/Origin1", "Xform", translation=[0.0, 0.0, 0.0])

    #Robot CONFIG 
    g1_cfg = G1_MINIMAL_CFG.copy()
    g1_cfg.prim_path = "/World/Origin1/Robot"
    robot = Articulation(cfg=g1_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())

    # Verify joint order matches the mapped file
    robot_joint_names = list(robot.data.joint_names)
    if robot_joint_names != joint_names_file:
        print("[ERROR] Joint order mismatch between mapped file and current G1 articulation.")
        print("[ERROR] File joints : ", joint_names_file)
        print("[ERROR] Robot joints: ", robot_joint_names)
        raise RuntimeError("Joint name/order mismatch. Rebuild the mapped file against this asset.")

    # Initial reset
    reset_robot(robot, default_joint_pos_np, args_cli.root_x, args_cli.root_y, args_cli.root_height)
    robot.update(sim.get_physics_dt())

    #Playback timing in simulation
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
    print("[INFO] This is physics-based tracking, not a balance policy.")

    frame_idx = 0
    step_count = 0

    while simulation_app.is_running():
        if step_count % sim_steps_per_frame == 0:
            q_target = torch.tensor(
                joint_targets_np[frame_idx],
                dtype=torch.float32,
                device=robot.device,
            ).unsqueeze(0)

            # Physics tracking path:
            # set desired joint positions then write to sim buffers then step physics later
            robot.set_joint_position_target(q_target)

            frame_idx += 1
            if frame_idx >= num_frames:
                if args_cli.loop:
                    frame_idx = 0
                    if args_cli.reset_on_loop:
                        reset_robot(robot, default_joint_pos_np, args_cli.root_x, args_cli.root_y, args_cli.root_height)
                else:
                    frame_idx = num_frames - 1

        # Key step for writing to the simulation as shown in the isaac sim docs.
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        step_count += 1


if __name__ == "__main__":
    import traceback
    try:
        print("[CHK] entering main")
        main()
        print("[CHK] main returned")
    except Exception:
        traceback.print_exc()
        input("Press Enter to close...")
        raise
    finally:
        simulation_app.close()
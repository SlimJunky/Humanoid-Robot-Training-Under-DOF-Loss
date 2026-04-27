
from __future__ import annotations

'''Take the robo-mimic policy checkpoint file and the BC demonstration json action bounds meta data then based off the rollout robomimic script
Take the motion policy and make it play repeatedly to gauge movement before RL tuning, AKA closed-loop BC policy player. ROllout the policy in Isaac Lab environment without
the need for the wrapper. Rebuilds same observation format used during BC training, ask learned policy for next action each sim step, converts that action back
into real joint targets and sends the targets to the Unitree G1 robot in the simulation.'''

'''Had to create a in WSL a way to export a plain PyTorch\TorchScript policy then load that exported file policy in this external project because Robomimic 
struggled to have even its simple packages work within the conda environment on Windows.'''

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

#Parser Arguments

parser = argparse.ArgumentParser(description="Closed-loop BC policy playback for G1 in Isaac Lab.")
parser.add_argument("bc_ckpt", type=str, help="Path to exported TorchScript BC policy .pt")
parser.add_argument("bc_meta_json", type=str, help="Path to BC dataset metadata json with action bounds")
parser.add_argument("--root-height", type=float, default=0.78)
parser.add_argument("--root-x", type=float, default=0.0)
parser.add_argument("--root-y", type=float, default=0.0)
parser.add_argument("--gait-period-s", type=float, default=4.25, help="Phase cycle period in seconds") # Mess around with phase gait for best start and cycle
parser.add_argument("--fall-reset-height", type=float, default=0.45, help="Reset if root height falls below this")
parser.add_argument("--debug-every", type=int, default=120, help="Print debug info every N sim steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

#Standard launch Minimal G1 IsaacLab
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets import G1_MINIMAL_CFG




# set joint positions, base root in isaac lab and resets articulation state so every reset puts robot in clean starting pose
def reset_robot(robot: Articulation, default_joint_pos_np: np.ndarray, root_x: float, root_y: float, root_z: float) -> None:
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


def build_proprio(q: np.ndarray, qd: np.ndarray, phase: float) -> np.ndarray:
    '''Build the same [q, qd, phase] observation used during BC training.'''
    return np.concatenate([q, qd, np.array([phase], dtype=np.float32)], axis=0).astype(np.float32)


def denormalize_actions(actions_norm: np.ndarray, action_lo: np.ndarray, action_hi: np.ndarray) -> np.ndarray:
    return (((actions_norm + 1.0) * 0.5) * (action_hi - action_lo) + action_lo).astype(np.float32)

# Debugging joint
def print_joint_sample(label: str, values: np.ndarray, joint_names: list[str], sample_names: list[str]) -> None:
    parts = []
    for name in sample_names:
        if name in joint_names:
            idx = joint_names.index(name)
            parts.append(f"{name}={values[idx]:+.4f}")
    print(f"[DBG] {label}: " + ", ".join(parts))


def main():
    ckpt_path = Path(args_cli.bc_ckpt) #.pth file of best BC model control policy
    meta_path = Path(args_cli.bc_meta_json) #BC demos dataset JSON metadata

    if not ckpt_path.exists():
        raise FileNotFoundError(f"BC checkpoint not found: {ckpt_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"BC metadata json not found: {meta_path}")
    
    # Debugging information
    print(f"[INFO] BC checkpoint: {ckpt_path.resolve()}")
    print(f"[INFO] BC metadata  : {meta_path.resolve()}")
    print(f"[INFO] Requested device: {args_cli.device}")

    #Load BC metadata required to determine the expected joint order and low / high action bounds used for normalized BC outputs
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    joint_names_file = meta["joint_names"]
    action_lo = np.asarray(meta["action_low"], dtype=np.float32)
    action_hi = np.asarray(meta["action_high"], dtype=np.float32)
    obs_dim = int(meta.get("obs_dim", 75))
    act_dim = int(meta.get("act_dim", len(joint_names_file)))
    

    print(f"[INFO] Loaded metadata joint count: {len(joint_names_file)}")
    print(f"[INFO] obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"[INFO] Action bounds shape: lo={action_lo.shape}, hi={action_hi.shape}")
    print(f"[INFO] Action low range : min={action_lo.min():+.4f}, max={action_lo.max():+.4f}")
    print(f"[INFO] Action high range: min={action_hi.min():+.4f}, max={action_hi.max():+.4f}")

    #Load torch policy model.
    policy = torch.jit.load(str(ckpt_path), map_location=args_cli.device)
    policy.eval()
    print("[INFO] TorchScript BC policy loaded successfully.")

    # Sanity check TorchScript from bugs making sure its right "shape" after processing.
    with torch.no_grad():
        test_obs = torch.zeros(1, obs_dim, dtype=torch.float32, device=args_cli.device)
        test_action = policy(test_obs)

        if isinstance(test_action, (tuple, list)):
            test_action = test_action[0]

        if not isinstance(test_action, torch.Tensor):
            raise TypeError(f"Policy output is not a torch.Tensor. Got: {type(test_action)}")

        print(f"[INFO] Policy test output shape: {tuple(test_action.shape)}")

        if test_action.ndim != 2:
            raise RuntimeError(f"Expected policy output shape [B, act_dim], got {tuple(test_action.shape)}")

        if test_action.shape[1] != act_dim:
            raise RuntimeError(
                f"Policy action dimension mismatch. Expected {act_dim}, got {test_action.shape[1]}"
            )

        if torch.isnan(test_action).any():
            raise RuntimeError("Policy produced NaN during test inference.")

    sim_cfg = SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.8, 2.0, 1.8], [0.0, 0.0, 0.8])

    print(f"[INFO] Simulation dt: {sim_cfg.dt:.6f}")
    print(f"[INFO] Gait period s: {args_cli.gait_period_s:.4f}")
    print(f"[INFO] Debug print every {args_cli.debug_every} sim steps")

    #Default for the simulation world
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sim_utils.create_prim("/World/Origin1", "Xform", translation=[0.0, 0.0, 0.0])

    g1_cfg = G1_MINIMAL_CFG.copy()
    g1_cfg.prim_path = "/World/Origin1/Robot"
    robot = Articulation(cfg=g1_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())

    # Joint order safety check making sure the G1 in Isaac Lab and G1 perceived from BC dataset have the right joint order.
    # BC model needs to output targets for right joints to follow correct motion policy behavior.
    robot_joint_names = list(robot.data.joint_names)
    print(f"[INFO] Robot joint count: {len(robot_joint_names)}")
    print(f"[INFO] First 10 robot joints: {robot_joint_names[:10]}")

    if robot_joint_names != joint_names_file:
        print("[ERROR] Joint order mismatch!")
        print(f"[ERROR] BC first 10 joints   : {joint_names_file[:10]}")
        print(f"[ERROR] Robot first 10 joints: {robot_joint_names[:10]}")
        raise RuntimeError(
            "Joint order mismatch between BC metadata and current G1 articulation.\n"
            f"BC:    {joint_names_file}\n"
            f"Robot: {robot_joint_names}"
        )
    
    print("[INFO] Joint order check passed.")

    # starting from all zero joint or robot default joint reset may be incorrect to match stable standing pose may need to tune for start.
    default_joint_pos_np = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)
    print(f"[INFO] Reset joint pose uses initial default robot pose. shape={default_joint_pos_np.shape}")

    reset_robot(robot, default_joint_pos_np, args_cli.root_x, args_cli.root_y, args_cli.root_height)
    robot.update(sim.get_physics_dt())
    print(f"[INFO] Initial reset at x={args_cli.root_x:.3f}, y={args_cli.root_y:.3f}, z={args_cli.root_height:.3f}")

    sim_dt = sim.get_physics_dt()
    phase = 0.0
    step_idx = 0
    reset_count = 0

    sample_joint_names = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "torso_joint",
    ]


    while simulation_app.is_running():
        # read current state live oservations from simulation physics
        q = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        qd = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)

        # reset on fall, fall detection so when playing BC control policy can be viewed without constantly resetting sim
        root_z = float(robot.data.root_pos_w[0, 2].item())
        if root_z < args_cli.fall_reset_height:
            print(f"[INFO] Resetting after fall. root_z={root_z:.3f}")
            print_joint_sample("q before fall", q, joint_names_file, sample_joint_names)
            print_joint_sample("qd before fall", qd, joint_names_file, sample_joint_names)
            reset_robot(robot, default_joint_pos_np, args_cli.root_x, args_cli.root_y, args_cli.root_height)
            robot.update(sim_dt)
            phase = 0.0
            continue

        # build policy observation. recreat observation format the policy expects
        obs_vec = build_proprio(q, qd, phase)
        # builds batch dimension that passes observation in robomimic policy
        #policy output is assumed as normalized 37-dim action vector.
        obs_tensor = torch.from_numpy(obs_vec[None, :]).to(robot.device)

        with torch.no_grad():
            action_norm = policy(obs_tensor)
            #robust to different TorchScript return styles
            if isinstance(action_norm, (tuple, list)):
                action_norm = action_norm[0]
            if not isinstance(action_norm, torch.Tensor):
                action_norm = torch.as_tensor(action_norm, device=robot.device)

            action_norm = action_norm.detach().cpu().numpy()[0]
        
        # Defensive clipping in normalized action space
        action_norm = np.clip(action_norm, -1.0, 1.0)


        # converts the normalized policy output used in robomimic BC dataset demo back into real joint target ranges.
        q_target = denormalize_actions(action_norm, action_lo, action_hi)
        q_target = np.clip(q_target, action_lo, action_hi)
        q_target_torch = torch.tensor(q_target, dtype=torch.float32, device=robot.device).unsqueeze(0)

        # Debug prints
        if step_idx == 0 or (args_cli.debug_every > 0 and step_idx % args_cli.debug_every == 0):
            root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
            root_lin_vel = robot.data.root_lin_vel_w[0].detach().cpu().numpy()

            print(
                f"[DBG] step={step_idx} phase={phase:.3f} root_z={root_z:.3f} "
                f"obs[min={obs_vec.min():+.4f}, max={obs_vec.max():+.4f}, mean={obs_vec.mean():+.4f}] "
                f"act_norm[min={action_norm.min():+.4f}, max={action_norm.max():+.4f}, mean={action_norm.mean():+.4f}] "
                f"q_target[min={q_target.min():+.4f}, max={q_target.max():+.4f}, mean={q_target.mean():+.4f}]"
            )
            print(
                f"[DBG] root_pos=({root_pos[0]:+.3f}, {root_pos[1]:+.3f}, {root_pos[2]:+.3f}) "
                f"root_lin_vel=({root_lin_vel[0]:+.3f}, {root_lin_vel[1]:+.3f}, {root_lin_vel[2]:+.3f})"
            )
            print_joint_sample("current q", q, joint_names_file, sample_joint_names)
            print_joint_sample("current qd", qd, joint_names_file, sample_joint_names)
            print_joint_sample("policy q_target", q_target, joint_names_file, sample_joint_names)


        #Apply targets and step sim
        robot.set_joint_position_target(q_target_torch)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim_dt)

        #Updates phase
        phase = (phase + sim_dt / args_cli.gait_period_s) % 1.0
        step_idx += 1


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("Press Enter to close...")
        raise
    finally:
        simulation_app.close()
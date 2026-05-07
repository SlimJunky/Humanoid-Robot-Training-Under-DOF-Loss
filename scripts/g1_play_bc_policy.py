
from __future__ import annotations

# Copyright (c) 2026, Mikolaj Wyrzykowski
# SPDX-License-Identifier: BSD-3-Clause

'''Take the robo-mimic policy checkpoint file and the BC demonstration json action bounds meta data then based off the rollout robomimic script
 make it play repeatedly to gauge movement before RL tuning, AKA closed-loop BC policy player. ROllout the policy in Isaac Lab environment without
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

# Classic helper find path roots
def find_project_root() -> Path:
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "source").exists():
            return parent
        if (parent / ".git").exists():
            return parent

    return current.parents[1]


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return project_root / path


def project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()

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
parser.add_argument("--control-decimation", type=int, default=2, help="Run policy every N physics steps. 2 means 120 Hz sim / 2 = 60 Hz policy control.")
# So I can compare walking policy to direct reference motion initially quickly
parser.add_argument("--" \
"use-reference-actions", action="store_true", help="Ignore BC policy and directly play mapped reference joint targets from source_mapped_npz.")
parser.add_argument("--freeze-root", action="store_true", help="Hold the pelvis/root fixed in space so the legs can be inspected without falling.")
parser.add_argument("--freeze-root-height", type=float, default=None, help="Root height used when --freeze-root is enabled. Defaults to --root-height.")



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

''' Keep the floating base fixed so only the joint motion is visually inspected. Do this for visual inspect of BC policy'''
def freeze_robot_root(robot: Articulation, root_x: float, root_y: float, root_z: float) -> None:
    root_state = robot.data.default_root_state.clone()

    root_state[:, 0] = root_x
    root_state[:, 1] = root_y
    root_state[:, 2] = root_z

    # Keep default orientation and zero root velocity.
    root_state[:, 7:] = 0.0

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])    


def build_proprio(q: np.ndarray, qd: np.ndarray, phase: float, include_phase: bool) -> np.ndarray:
    '''Build the same [q, qd, phase] observation used during BC training.'''
    pieces = [q.astype(np.float32), qd.astype(np.float32)]

    if include_phase:
        pieces.append(np.array([phase], dtype=np.float32))

    return np.concatenate(pieces, axis=0).astype(np.float32)


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
    project_root = find_project_root()

    ckpt_path = resolve_project_path(args_cli.bc_ckpt.strip(), project_root) #.pth file of best BC model control policy
    meta_path = resolve_project_path(args_cli.bc_meta_json.strip(), project_root) #BC demos dataset JSON metadata

    if not ckpt_path.exists():
        raise FileNotFoundError(f"BC checkpoint not found: {ckpt_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"BC metadata json not found: {meta_path}")
    
    # Debugging information
    print(f"BC checkpoint: {project_relative(ckpt_path, project_root)}")
    print(f"BC metadata  : {project_relative(meta_path, project_root)}")
    print(f"Requested device: {args_cli.device}")

    #Load BC metadata required to determine the expected joint order and low / high action bounds used for normalized BC outputs
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    joint_names_file = meta["joint_names"]
    action_lo = np.asarray(meta["action_low"], dtype=np.float32)
    action_hi = np.asarray(meta["action_high"], dtype=np.float32)

    obs_dim = int(meta.get("obs_dim", 75))
    act_dim = int(meta.get("act_dim", len(joint_names_file)))

    include_phase = bool(meta.get("include_phase", obs_dim == 75))
    actions_normalized = bool(meta.get("actions_normalized", True))
    

    print(f"Loaded metadata joint count: {len(joint_names_file)}")
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"Action bounds shape: lo={action_lo.shape}, hi={action_hi.shape}")
    print(f"Action low range : min={action_lo.min():+.4f}, max={action_lo.max():+.4f}")
    print(f"Action high range: min={action_hi.min():+.4f}, max={action_hi.max():+.4f}")
    print(f"include_phase={include_phase}")
    print(f"actions_normalized={actions_normalized}")

    if action_lo.shape[0] != act_dim or action_hi.shape[0] != act_dim:
        raise RuntimeError(
            f"Action bounds dimension mismatch. "
            f"act_dim={act_dim}, action_lo={action_lo.shape}, action_hi={action_hi.shape}"
        )

    policy = torch.jit.load(str(ckpt_path), map_location=args_cli.device)
    policy.eval()
    print("TorchScript BC policy loaded successfully.")



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

    print(f"Simulation dt: {sim_cfg.dt:.6f}")
    print(f"Gait period s: {args_cli.gait_period_s:.4f}")
    print(f"Debug print every {args_cli.debug_every} sim steps")

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
    print(f"Robot joint count: {len(robot_joint_names)}")
    print(f"First 10 robot joints: {robot_joint_names[:10]}")

    if robot_joint_names != joint_names_file:
        print("[ERROR] Joint order mismatch!")
        print(f"[ERROR] BC first 10 joints   : {joint_names_file[:10]}")
        print(f"[ERROR] Robot first 10 joints: {robot_joint_names[:10]}")

        raise RuntimeError(
            "joint order mismatch between BC metadata and current G1 articulation.\n"
            f"BC:    {joint_names_file}\n"
            f"Robot: {robot_joint_names}"
        )
    
    print("Joint order check passed.")

    # starting from all zero joint or robot default joint reset may be incorrect to match stable standing pose may need to tune for start.

    source_mapped_npz_value = meta.get("source_mapped_npz", "")

    if not source_mapped_npz_value:
        raise KeyError(
            "BC metadata JSON is missing 'source_mapped_npz'. "
            "Add a project-relative path such as "
            "'data/mapped/Walk/37_01_poses_slow_walk_retarget_ready_g1_first_pass.npz'."
        )

    source_mapped_npz = resolve_project_path(meta["source_mapped_npz"], project_root)

    # Should put this to force mapped .npz to be available 
    if not source_mapped_npz.exists():
        raise FileNotFoundError(f"source_mapped_npz from metadata was not found: {source_mapped_npz}")


    if source_mapped_npz.exists():
        with np.load(source_mapped_npz, allow_pickle=True) as data:
            reference_joint_targets = np.asarray(data["joint_targets"], dtype=np.float32)
            T_ref = reference_joint_targets.shape[0]

        if reference_joint_targets.ndim != 2:
            raise RuntimeError(f"Expected reference joint_targets shape [T, J], got {reference_joint_targets.shape}")

        if reference_joint_targets.shape[1] != len(robot_joint_names):
            raise RuntimeError(
                f"Reference joint target dimension mismatch. "
                f"Mapped file has {reference_joint_targets.shape[1]}, robot has {len(robot_joint_names)}"
            )

        default_joint_pos_np = reference_joint_targets[0].copy()
        print("Reset joint pose uses first frame of mapped reference motion.")

    else:
        default_joint_pos_np = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)
        print("could not find source_mapped_npz. Using default Isaac Lab joint pose.")


    #default_joint_pos_np = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)
    #print(f"Reset joint pose uses initial default robot pose. shape={default_joint_pos_np.shape}")

    reset_robot(robot, default_joint_pos_np, args_cli.root_x, args_cli.root_y, args_cli.root_height)
    robot.update(sim.get_physics_dt())
    print(f"Initial reset at x={args_cli.root_x:.3f}, y={args_cli.root_y:.3f}, z={args_cli.root_height:.3f}")

    sim_dt = sim.get_physics_dt()
    phase = 0.0
    step_idx = 0
    reset_count = 0

    control_decimation = max(1, int(args_cli.control_decimation))

    last_q_target_torch = None
    last_q_target = default_joint_pos_np.copy()
    last_policy_out = np.zeros((act_dim,), dtype=np.float32)

    print(f"Control decimation: {control_decimation}")
    print(f"Effective policy rate: {1.0 / (sim_dt * control_decimation):.2f} Hz") # with dt = 1/ 120 then control decimation policy rate of 60.00 Hz

    sample_joint_names = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "torso_joint",
    ]

    lower_body_joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]

    lower_body_joint_ids = np.array( [joint_names_file.index(name) for name in lower_body_joint_names if name in joint_names_file], dtype=np.int64,)

    left_foot_ids, left_foot_names = robot.find_bodies(".*left_ankle_roll.*")
    right_foot_ids, right_foot_names = robot.find_bodies(".*right_ankle_roll.*")

    if len(left_foot_ids) == 0 or len(right_foot_ids) == 0:
        raise RuntimeError("Could not find left/right ankle roll bodies for debug.")

    left_foot_body_id = left_foot_ids[0]
    right_foot_body_id = right_foot_ids[0]

    print(f"Debug left foot body : {left_foot_names[0]} id={left_foot_body_id}")
    print(f"Debug right foot body: {right_foot_names[0]} id={right_foot_body_id}")

    cycle_idx = 0
    prev_phase = phase

    cycle_stats = {
        "bc_ref_mean": [],
        "bc_ref_max": [],
        "q_ref_mean": [],
        "lower_bc_ref_mean": [],
        "lower_q_ref_mean": [],
        "target_jump_mean": [],
        "policy_sat_frac": [],
        "left_foot_x": [],
        "right_foot_x": [],
        "left_foot_z": [],
        "right_foot_z": [],
        "foot_x_gap": [],
        "foot_z_diff": [],
    }


    while simulation_app.is_running():
        # read current state live oservations from simulation physics
        q = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        qd = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)

        # reset on fall, fall detection so when playing BC control policy can be viewed without constantly resetting sim
        root_z = float(robot.data.root_pos_w[0, 2].item())


        if root_z < args_cli.fall_reset_height:
            print(f"Resetting after fall. root_z={root_z:.3f}")
            print_joint_sample("q before fall", q, joint_names_file, sample_joint_names)
            print_joint_sample("qd before fall", qd, joint_names_file, sample_joint_names)
            reset_robot(robot, default_joint_pos_np, args_cli.root_x, args_cli.root_y, args_cli.root_height)
            robot.update(sim_dt)
            phase = 0.0
            step_idx = 0
            reset_count += 1

            last_q_target_torch = None
            last_q_target = default_joint_pos_np.copy()
            last_policy_out = np.zeros((act_dim,), dtype=np.float32)
            continue

        # build policy observation. recreate observation format the policy expects
        # builds batch dimension that passes observation in robomimic policy
        # policy output is assumed as normalized 37-dim action vector.

        # Decimated policy inference.
        ''' step 0: run policy, compute new target
           step 1: hold previous target
           step 2: run policy, compute new target
           step 3: hold previous target
        # This better matches the training data rate. '''

        should_update_policy = step_idx % control_decimation == 0 or last_q_target_torch is None

        if should_update_policy:
            obs_vec = build_proprio(q, qd, phase, include_phase)

            if obs_vec.shape[0] != obs_dim:
                raise RuntimeError(
                    f"Observation dimension mismatch. Built {obs_vec.shape[0]}, expected {obs_dim}."
                )

            if args_cli.use_reference_actions:
                # Direct reference playback:
                # skip the BC policy and directly command the next mapped reference frame.
                ref_idx = int(round(phase * T_ref)) % T_ref
                ref_next_idx = (ref_idx + 1) % T_ref

                q_target = reference_joint_targets[ref_next_idx].copy().astype(np.float32)
                q_target = np.clip(q_target, action_lo, action_hi)

                # Only used for debug printing. This is not a real policy output in this mode.
                policy_out = np.zeros((act_dim,), dtype=np.float32)

            else:
                # Normal BC policy playback.
                obs_tensor = torch.from_numpy(obs_vec[None, :]).to(robot.device)

                with torch.no_grad():
                    policy_out = policy(obs_tensor)

                    if isinstance(policy_out, (tuple, list)):
                        policy_out = policy_out[0]

                    if not isinstance(policy_out, torch.Tensor):
                        policy_out = torch.as_tensor(policy_out, device=robot.device)

                    policy_out = policy_out.detach().cpu().numpy()[0].astype(np.float32)

                if policy_out.shape[0] != act_dim:
                    raise RuntimeError(
                        f"Policy action dimension mismatch. Got {policy_out.shape[0]}, expected {act_dim}."
                    )

                if actions_normalized:
                    policy_out = np.clip(policy_out, -1.0, 1.0)
                    q_target = denormalize_actions(policy_out, action_lo, action_hi)
                else:
                    q_target = policy_out.astype(np.float32)

                q_target = np.clip(q_target, action_lo, action_hi)

            last_policy_out = policy_out.copy()
            last_q_target = q_target.copy()
            last_q_target_torch = torch.tensor(q_target,dtype=torch.float32, device=robot.device,).unsqueeze(0)


        else:
            # Hold previous target during intermediate physics step.
            obs_vec = build_proprio(q, qd, phase, include_phase)
            policy_out = last_policy_out
            q_target = last_q_target

        q_target_torch = last_q_target_torch


        # Debug prints
        if step_idx == 0 or (args_cli.debug_every > 0 and step_idx % args_cli.debug_every == 0):
            root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
            root_lin_vel = robot.data.root_lin_vel_w[0].detach().cpu().numpy()

            target_jump = float(np.max(np.abs(q_target - q)))

            ref_idx = int(round(phase * T_ref)) % T_ref
            ref_next_idx = (ref_idx + 1) % T_ref
            ref_next_q = reference_joint_targets[ref_next_idx]

            ref_error = float(np.mean(np.abs(q_target - ref_next_q)))
            ref_max_error = float(np.max(np.abs(q_target - ref_next_q)))

            ref_current_q = reference_joint_targets[ref_idx]

            q_ref_error = float(np.mean(np.abs(q - ref_current_q)))
            lower_bc_ref_error = float(np.mean(np.abs(q_target[lower_body_joint_ids] - ref_next_q[lower_body_joint_ids])))
            lower_q_ref_error = float(np.mean(np.abs(q[lower_body_joint_ids] - ref_current_q[lower_body_joint_ids])))

            policy_sat_frac = float(np.mean(np.abs(policy_out) > 0.98))

            left_foot_pos = robot.data.body_pos_w[0, left_foot_body_id].detach().cpu().numpy()
            right_foot_pos = robot.data.body_pos_w[0, right_foot_body_id].detach().cpu().numpy()
            root_pos_dbg = robot.data.root_pos_w[0].detach().cpu().numpy()

            left_foot_rel = left_foot_pos - root_pos_dbg
            right_foot_rel = right_foot_pos - root_pos_dbg

            foot_x_gap = float(abs(left_foot_rel[0] - right_foot_rel[0]))
            foot_z_diff = float(abs(left_foot_rel[2] - right_foot_rel[2]))

            #Log for reference of gait cycle compared to mapped .npz motion
            cycle_stats["bc_ref_mean"].append(ref_error)
            cycle_stats["bc_ref_max"].append(ref_max_error)
            cycle_stats["q_ref_mean"].append(q_ref_error)
            cycle_stats["lower_bc_ref_mean"].append(lower_bc_ref_error)
            cycle_stats["lower_q_ref_mean"].append(lower_q_ref_error)
            cycle_stats["target_jump_mean"].append(target_jump)
            cycle_stats["policy_sat_frac"].append(policy_sat_frac)
            cycle_stats["left_foot_x"].append(float(left_foot_rel[0]))
            cycle_stats["right_foot_x"].append(float(right_foot_rel[0]))
            cycle_stats["left_foot_z"].append(float(left_foot_rel[2]))
            cycle_stats["right_foot_z"].append(float(right_foot_rel[2]))
            cycle_stats["foot_x_gap"].append(foot_x_gap)
            cycle_stats["foot_z_diff"].append(foot_z_diff)

            print(
                f"[DBG] step={step_idx} phase={phase:.3f} root_z={root_z:.3f} "
                f"update_policy={should_update_policy} "
                f"ref_idx={ref_idx} ref_next_idx={ref_next_idx} "
                f"mean|q_target-ref_next|={ref_error:.4f} "
                f"max|q_target-ref_next|={ref_max_error:.4f} "
                f"obs[min={obs_vec.min():+.4f}, max={obs_vec.max():+.4f}, mean={obs_vec.mean():+.4f}] "
                f"policy_out[min={policy_out.min():+.4f}, max={policy_out.max():+.4f}, mean={policy_out.mean():+.4f}] "
                f"q_target[min={q_target.min():+.4f}, max={q_target.max():+.4f}, mean={q_target.mean():+.4f}] "
                f"max|q_target-q|={target_jump:.4f}"
            )

            print(
                f"[DBG] root_pos=({root_pos[0]:+.3f}, {root_pos[1]:+.3f}, {root_pos[2]:+.3f}) "
                f"root_lin_vel=({root_lin_vel[0]:+.3f}, {root_lin_vel[1]:+.3f}, {root_lin_vel[2]:+.3f})"
            )

            print(
                f"[DBG] tracking: "
                f"mean|q-ref_current|={q_ref_error:.4f} "
                f"lower_mean|q_target-ref_next|={lower_bc_ref_error:.4f} "
                f"lower_mean|q-ref_current|={lower_q_ref_error:.4f} "
                f"policy_sat_frac={policy_sat_frac:.3f}"
            )

            print(
                f"[DBG] feet_rel_to_root: "
                f"L(x={left_foot_rel[0]:+.3f}, y={left_foot_rel[1]:+.3f}, z={left_foot_rel[2]:+.3f}) "
                f"R(x={right_foot_rel[0]:+.3f}, y={right_foot_rel[1]:+.3f}, z={right_foot_rel[2]:+.3f}) "
                f"foot_x_gap={foot_x_gap:.3f} "
                f"foot_z_diff={foot_z_diff:.3f}"
            )

            print_joint_sample("current q", q, joint_names_file, sample_joint_names)
            print_joint_sample("current qd", qd, joint_names_file, sample_joint_names)
            print_joint_sample("policy q_target", q_target, joint_names_file, sample_joint_names)

            #More debugging to see what the policy is trying to do with the joints next
            print_joint_sample("target delta", q_target - q, joint_names_file, sample_joint_names)



        #Apply targets and step sim

        if args_cli.freeze_root:
            freeze_z = args_cli.freeze_root_height if args_cli.freeze_root_height is not None else args_cli.root_height
            freeze_robot_root(robot, args_cli.root_x, args_cli.root_y, freeze_z)

        robot.set_joint_position_target(q_target_torch)
        robot.write_data_to_sim()

        sim.step()

        if args_cli.freeze_root:
            freeze_z = args_cli.freeze_root_height if args_cli.freeze_root_height is not None else args_cli.root_height
            freeze_robot_root(robot, args_cli.root_x, args_cli.root_y, freeze_z)
        robot.update(sim_dt)

        #Updates phase
        if should_update_policy:
            phase = (phase + (sim_dt * control_decimation) / args_cli.gait_period_s) % 1.0


        '''Summary cycle block for debugging and understanding how BC prior is predicting next movements in cycle'''
        if phase < prev_phase and len(cycle_stats["bc_ref_mean"]) > 0:
            print("\n================ BC / REFERENCE CYCLE SUMMARY ================")
            print(f"[CYCLE {cycle_idx}] frozen_root={args_cli.freeze_root} use_reference_actions={args_cli.use_reference_actions}")
            print(
                f"[CYCLE {cycle_idx}] bc_to_ref: "
                f"mean={np.mean(cycle_stats['bc_ref_mean']):.5f}, "
                f"max_mean={np.max(cycle_stats['bc_ref_mean']):.5f}, "
                f"max_abs={np.max(cycle_stats['bc_ref_max']):.5f}"
            )
            print(
                f"[CYCLE {cycle_idx}] sim_tracking: "
                f"mean|q-ref_current|={np.mean(cycle_stats['q_ref_mean']):.5f}, "
                f"lower_mean|q-ref_current|={np.mean(cycle_stats['lower_q_ref_mean']):.5f}, "
                f"lower_mean|q_target-ref_next|={np.mean(cycle_stats['lower_bc_ref_mean']):.5f}"
            )
            print(
                f"[CYCLE {cycle_idx}] policy: "
                f"mean_target_jump={np.mean(cycle_stats['target_jump_mean']):.5f}, "
                f"max_target_jump={np.max(cycle_stats['target_jump_mean']):.5f}, "
                f"mean_policy_sat_frac={np.mean(cycle_stats['policy_sat_frac']):.3f}"
            )
            print(
                f"[CYCLE {cycle_idx}] foot_motion_rel_to_root: "
                f"Lx_range=({np.min(cycle_stats['left_foot_x']):+.3f}, {np.max(cycle_stats['left_foot_x']):+.3f}), "
                f"Rx_range=({np.min(cycle_stats['right_foot_x']):+.3f}, {np.max(cycle_stats['right_foot_x']):+.3f}), "
                f"Lz_range=({np.min(cycle_stats['left_foot_z']):+.3f}, {np.max(cycle_stats['left_foot_z']):+.3f}), "
                f"Rz_range=({np.min(cycle_stats['right_foot_z']):+.3f}, {np.max(cycle_stats['right_foot_z']):+.3f})"
            )
            print(
                f"[CYCLE {cycle_idx}] foot_gap: "
                f"mean_x_gap={np.mean(cycle_stats['foot_x_gap']):.3f}, "
                f"max_x_gap={np.max(cycle_stats['foot_x_gap']):.3f}, "
                f"mean_z_diff={np.mean(cycle_stats['foot_z_diff']):.3f}, "
                f"max_z_diff={np.max(cycle_stats['foot_z_diff']):.3f}"
            )
            print("==============================================================\n")

            cycle_idx += 1
            for key in cycle_stats:
                cycle_stats[key].clear()

        prev_phase = phase
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
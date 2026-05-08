from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args


parser = argparse.ArgumentParser(description="Evaluate G1 PPO walking policy with clean metric CSV logging.")

# Normal RSL-RL / Isaac Lab args
parser.add_argument("--task", type=str, default="Isaac-G1-BC-PPO-Walk-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--out_csv", type=str, default="results/g1_policy_eval.csv")
parser.add_argument("--robot_name", type=str, default="robot")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--debug_interval_s", type=float, default=0.5)

# Fault args
parser.add_argument("--fault_mode", type=str, default="none", choices=["none", "torque", "lock"])
parser.add_argument("--fault_joint", type=str, default="")
parser.add_argument("--fault_time_s", type=float, default=2.0)
parser.add_argument("--torque_scale", type=float, default=1.0, help="0.5 = 50 percent torque, 0.0 = no torque")
parser.add_argument("--lock_epsilon", type=float, default=0.001, help="Small joint-limit band around locked angle")

# Contact / metric heuristics
parser.add_argument("--foot_contact_height", type=float, default=0.08)
parser.add_argument("--fall_height", type=float, default=0.45)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("This metric script is intentionally designed for --num_envs 1 for clean per-episode CSV results.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import math
import os
import statistics
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg




CSV_FIELDS = [
    "task",
    "checkpoint",
    "episode",
    "fault_mode",
    "fault_joint",
    "fault_time_s",
    "torque_scale",
    "fault_applied",
    "fault_applied_time_s",
    "fault_lock_angle_rad",
    "success_no_fall",
    "timeout",
    "fall",
    "episode_duration_s",
    "time_to_fall_s",
    "forward_distance_m",
    "final_x_m",
    "final_y_m",
    "max_lateral_drift_m",
    "mean_forward_vel_b_mps",
    "max_forward_vel_b_mps",
    "mean_abs_lateral_vel_b_mps",
    "mean_root_height_m",
    "min_root_height_m",
    "mean_tilt_metric",
    "max_tilt_metric",
    "mean_base_ang_vel_norm",
    "max_base_ang_vel_norm",
    "left_foot_clearance_mean_m",
    "left_foot_clearance_max_m",
    "right_foot_clearance_mean_m",
    "right_foot_clearance_max_m",
    "left_contact_fraction",
    "right_contact_fraction",
    "left_contact_transitions",
    "right_contact_transitions",
    "left_knee_angle_mean_rad",
    "left_knee_angle_max_rad",
    "right_knee_angle_mean_rad",
    "right_knee_angle_max_rad",
    "distance_before_fault_m",
    "distance_after_fault_m",
    "mean_forward_vel_before_fault_mps",
    "mean_forward_vel_after_fault_mps",
    "mean_root_height_after_fault_m",
    "min_root_height_after_fault_m",
    "max_tilt_after_fault",
    "mean_action_norm",
    "max_action_norm",
    "mean_action_rate",
    "fault_joint_pos_at_fault_rad",
    "fault_joint_pos_final_rad",
    "fault_joint_vel_mean_after_fault_radps",
    "fault_joint_vel_max_after_fault_radps",
]


def fmean(xs, default=""):
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return statistics.fmean(xs) if xs else default


def fmax(xs, default=""):
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return max(xs) if xs else default


def fmin(xs, default=""):
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return min(xs) if xs else default


def norm_name(s: str) -> str:
    return s.lower().replace("_joint", "").replace("_link", "").replace("-", "_")


def find_index(names: list[str], query: str, required: bool = True) -> int | None:
    if not query:
        if required:
            raise ValueError("Empty name query.")
        return None

    q = norm_name(query)

    for i, name in enumerate(names):
        if norm_name(name) == q:
            return i

    for i, name in enumerate(names):
        if q in norm_name(name):
            return i

    if required:
        print("\n[ERROR] Could not find:", query)
        print("[INFO] Available names:")
        for name in names:
            print("  ", name)
        raise ValueError(f"Could not find index for '{query}'")
    return None


def find_first_index(names: list[str], candidates: list[str]) -> int | None:
    for c in candidates:
        idx = find_index(names, c, required=False)
        if idx is not None:
            return idx
    return None


def get_scene_robot(base_env, robot_name: str):
    if robot_name in base_env.scene:
        return base_env.scene[robot_name]

    # fallback: find first asset with joint_pos/root_pos data
    for name, asset in base_env.scene.items():
        if hasattr(asset, "data") and hasattr(asset.data, "joint_pos") and hasattr(asset.data, "root_pos_w"):
            print(f"robot_name='{robot_name}' not found. Using scene asset '{name}' instead.")
            return asset

    raise RuntimeError(f"Could not find robot asset '{robot_name}' in env.scene.")


def get_step_dt(base_env) -> float:
    if hasattr(base_env, "step_dt"):
        return float(base_env.step_dt)

    cfg = base_env.cfg
    return float(cfg.sim.dt * cfg.decimation)


def get_episode_length_s(base_env) -> float:
    if hasattr(base_env.cfg, "episode_length_s"):
        return float(base_env.cfg.episode_length_s)

    if hasattr(base_env, "max_episode_length"):
        return float(base_env.max_episode_length * get_step_dt(base_env))

    return 8.0


def get_names(robot):
    joint_names = list(getattr(robot, "joint_names", []))
    body_names = list(getattr(robot, "body_names", []))

    if not joint_names and hasattr(robot.data, "joint_names"):
        joint_names = list(robot.data.joint_names)
    if not body_names and hasattr(robot.data, "body_names"):
        body_names = list(robot.data.body_names)

    if not joint_names:
        raise RuntimeError("Could not read robot joint_names.")
    if not body_names:
        raise RuntimeError("Could not read robot body_names.")

    return joint_names, body_names


def reset_env_get_obs(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def unwrap_scalar_bool(x) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(x.detach().flatten()[0].item())
    return bool(x)


def get_timeout_from_extras(extras, episode_time_s: float, episode_length_s: float, done: bool) -> bool:
    if isinstance(extras, dict):
        for key in ["time_outs", "timeouts", "TimeLimit.truncated"]:
            if key in extras:
                return unwrap_scalar_bool(extras[key])
    return bool(done and episode_time_s >= episode_length_s - 1.5e-3)


def append_csv(path: str, row: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(path).exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


class EpisodeMetricBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t = []
        self.root_x = []
        self.root_y = []
        self.root_z = []
        self.vx_b = []
        self.vy_b = []
        self.tilt = []
        self.ang_vel_norm = []
        self.left_foot_z = []
        self.right_foot_z = []
        self.left_contact = []
        self.right_contact = []
        self.left_knee = []
        self.right_knee = []
        self.start_x = None
        self.start_y = None
        self.action_norm = []
        self.action_rate = []
        self.prev_action = None
        self.fault_joint_pos = []
        self.fault_joint_vel = []
        self.fault_joint_pos_at_fault = None

    def update(
        self,
        t_s: float,
        robot,
        left_foot_body_id: int | None,
        right_foot_body_id: int | None,
        left_knee_joint_id: int | None,
        right_knee_joint_id: int | None,
        foot_contact_height: float,
    ):
        root_pos = robot.data.root_pos_w[0].detach()
        root_lin_vel_b = robot.data.root_lin_vel_b[0].detach()
        root_ang_vel_b = robot.data.root_ang_vel_b[0].detach()

        if hasattr(robot.data, "projected_gravity_b"):
            projected_gravity_b = robot.data.projected_gravity_b[0].detach()
            tilt_metric = torch.linalg.norm(projected_gravity_b[:2]).item()
        else:
            tilt_metric = float("nan")

        x = float(root_pos[0].item())
        y = float(root_pos[1].item())
        z = float(root_pos[2].item())

        if self.start_x is None:
            self.start_x = x
            self.start_y = y

        self.t.append(float(t_s))
        self.root_x.append(x)
        self.root_y.append(y)
        self.root_z.append(z)
        self.vx_b.append(float(root_lin_vel_b[0].item()))
        self.vy_b.append(float(root_lin_vel_b[1].item()))
        self.tilt.append(float(tilt_metric))
        self.ang_vel_norm.append(float(torch.linalg.norm(root_ang_vel_b).item()))

        body_pos = robot.data.body_pos_w[0].detach()

        if left_foot_body_id is not None:
            lz = float(body_pos[left_foot_body_id, 2].item())
            self.left_foot_z.append(lz)
            self.left_contact.append(1 if lz <= foot_contact_height else 0)

        if right_foot_body_id is not None:
            rz = float(body_pos[right_foot_body_id, 2].item())
            self.right_foot_z.append(rz)
            self.right_contact.append(1 if rz <= foot_contact_height else 0)

        joint_pos = robot.data.joint_pos[0].detach()

        if left_knee_joint_id is not None:
            self.left_knee.append(float(joint_pos[left_knee_joint_id].item()))

        if right_knee_joint_id is not None:
            self.right_knee.append(float(joint_pos[right_knee_joint_id].item()))

    @staticmethod
    def contact_transitions(contact_list: list[int]) -> int:
        if len(contact_list) < 2:
            return 0
        return sum(1 for a, b in zip(contact_list[:-1], contact_list[1:]) if a != b)

    def finalize(
        self,
        task: str,
        checkpoint: str,
        episode: int,
        fault_mode: str,
        fault_joint: str,
        fault_time_s: float,
        torque_scale: float,
        fault_applied: bool,
        fault_applied_time_s,
        fault_lock_angle,
        timeout: bool,
        fall: bool,
    ) -> dict:
        duration = self.t[-1] if self.t else 0.0
        final_x = self.root_x[-1] if self.root_x else ""
        final_y = self.root_y[-1] if self.root_y else ""

        forward_distance = ""
        max_lateral_drift = ""

        if self.root_x and self.start_x is not None:
            forward_distance = self.root_x[-1] - self.start_x

        if self.root_y and self.start_y is not None:
            max_lateral_drift = max(abs(y - self.start_y) for y in self.root_y)

        return {
            "task": task,
            "checkpoint": checkpoint,
            "episode": episode,
            "fault_mode": fault_mode,
            "fault_joint": fault_joint,
            "fault_time_s": fault_time_s,
            "torque_scale": torque_scale,
            "fault_applied": int(fault_applied),
            "fault_applied_time_s": fault_applied_time_s if fault_applied_time_s is not None else "",
            "fault_lock_angle_rad": fault_lock_angle if fault_lock_angle is not None else "",
            "success_no_fall": int(not fall),
            "timeout": int(timeout),
            "fall": int(fall),
            "episode_duration_s": duration,
            "time_to_fall_s": duration if fall else "",
            "forward_distance_m": forward_distance,
            "final_x_m": final_x,
            "final_y_m": final_y,
            "max_lateral_drift_m": max_lateral_drift,
            "mean_forward_vel_b_mps": fmean(self.vx_b),
            "max_forward_vel_b_mps": fmax(self.vx_b),
            "mean_abs_lateral_vel_b_mps": fmean([abs(v) for v in self.vy_b]),
            "mean_root_height_m": fmean(self.root_z),
            "min_root_height_m": fmin(self.root_z),
            "mean_tilt_metric": fmean(self.tilt),
            "max_tilt_metric": fmax(self.tilt),
            "mean_base_ang_vel_norm": fmean(self.ang_vel_norm),
            "max_base_ang_vel_norm": fmax(self.ang_vel_norm),
            "left_foot_clearance_mean_m": fmean(self.left_foot_z),
            "left_foot_clearance_max_m": fmax(self.left_foot_z),
            "right_foot_clearance_mean_m": fmean(self.right_foot_z),
            "right_foot_clearance_max_m": fmax(self.right_foot_z),
            "left_contact_fraction": fmean(self.left_contact),
            "right_contact_fraction": fmean(self.right_contact),
            "left_contact_transitions": self.contact_transitions(self.left_contact),
            "right_contact_transitions": self.contact_transitions(self.right_contact),
            "left_knee_angle_mean_rad": fmean(self.left_knee),
            "left_knee_angle_max_rad": fmax(self.left_knee),
            "right_knee_angle_mean_rad": fmean(self.right_knee),
            "right_knee_angle_max_rad": fmax(self.right_knee),
        }


class FaultController:
    def __init__(self, robot, joint_names: list[str], args):
        self.robot = robot
        self.args = args
        self.device = robot.device
        self.joint_id = None
        self.original_effort_limits = None
        self.original_pos_limits = None
        self.original_vel_limits = None

        if args.fault_mode != "none":
            self.joint_id = find_index(joint_names, args.fault_joint, required=True)
            print(f"[INFO] Fault joint '{args.fault_joint}' resolved to index {self.joint_id}: {joint_names[self.joint_id]}")

        if hasattr(robot.data, "joint_effort_limits"):
            self.original_effort_limits = robot.data.joint_effort_limits.clone()
        if hasattr(robot.data, "joint_pos_limits"):
            self.original_pos_limits = robot.data.joint_pos_limits.clone()
        elif hasattr(robot.data, "soft_joint_pos_limits"):
            self.original_pos_limits = robot.data.soft_joint_pos_limits.clone()
        if hasattr(robot.data, "joint_vel_limits"):
            self.original_vel_limits = robot.data.joint_vel_limits.clone()

    def restore(self):
        if self.joint_id is None:
            return

        jid = [self.joint_id]

        if self.original_effort_limits is not None:
            effort = self.original_effort_limits[:, jid]
            self.robot.write_joint_effort_limit_to_sim(effort, joint_ids=jid)

        if self.original_pos_limits is not None and hasattr(self.robot, "write_joint_position_limit_to_sim"):
            pos_lim = self.original_pos_limits[:, jid, :]
            self.robot.write_joint_position_limit_to_sim(pos_lim, joint_ids=jid, warn_limit_violation=False)

        if self.original_vel_limits is not None:
            vel_lim = self.original_vel_limits[:, jid]
            self.robot.write_joint_velocity_limit_to_sim(vel_lim, joint_ids=jid)

    def apply_if_needed(self, t_s: float, already_applied: bool):
        if self.args.fault_mode == "none" or already_applied:
            return already_applied, None

        if t_s < self.args.fault_time_s:
            return already_applied, None

        jid = [self.joint_id]

        if self.args.fault_mode == "torque":
            if self.original_effort_limits is None:
                raise RuntimeError("robot.data.joint_effort_limits not available, cannot apply torque fault.")

            scaled = self.original_effort_limits[:, jid] * float(self.args.torque_scale)
            self.robot.write_joint_effort_limit_to_sim(scaled, joint_ids=jid)

            print(
                f"[FAULT] t={t_s:.3f}s torque scale applied to {self.args.fault_joint}: "
                f"{self.args.torque_scale:.3f}"
            )
            return True, None

        if self.args.fault_mode == "lock":
            if not hasattr(self.robot, "write_joint_position_limit_to_sim"):
                raise RuntimeError("robot.write_joint_position_limit_to_sim not available, cannot apply lock fault.")

            lock_angle = float(self.robot.data.joint_pos[0, self.joint_id].detach().item())
            eps = float(self.args.lock_epsilon)

            limits = torch.tensor([[[lock_angle - eps, lock_angle + eps]]], device=self.device)
            self.robot.write_joint_position_limit_to_sim(limits, joint_ids=jid, warn_limit_violation=False)

            # Restrict velocity as well to make the lock more obvious.
            if hasattr(self.robot, "write_joint_velocity_limit_to_sim"):
                vel_limit = torch.zeros((1, 1), device=self.device) + 0.01
                self.robot.write_joint_velocity_limit_to_sim(vel_limit, joint_ids=jid)

            print(
                f"[FAULT] t={t_s:.3f}s joint lock applied to {self.args.fault_joint}: "
                f"angle={lock_angle:.5f} rad"
            )
            return True, lock_angle

        return already_applied, None


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    checkpoint_arg = str(args_cli.checkpoint) if args_cli.checkpoint else ""
    if checkpoint_arg and Path(checkpoint_arg).is_file():
        resume_path = str(Path(checkpoint_arg).resolve())
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"Loading checkpoint: {resume_path}")
    print(f"Writing CSV results to: {args_cli.out_csv}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = env.unwrapped

    robot = get_scene_robot(base_env, args_cli.robot_name)
    joint_names, body_names = get_names(robot)

    print("Available joint names:")
    for j in joint_names:
        print("  ", j)

    left_foot_body_id = find_first_index(
        body_names,
        ["left_ankle_roll", "left_foot", "left_toe", "left_sole", "left_ankle"],
    )
    right_foot_body_id = find_first_index(
        body_names,
        ["right_ankle_roll", "right_foot", "right_toe", "right_sole", "right_ankle"],
    )

    left_knee_joint_id = find_first_index(joint_names, ["left_knee"])
    right_knee_joint_id = find_first_index(joint_names, ["right_knee"])

    print(f"left_foot_body_id={left_foot_body_id}")
    print(f"right_foot_body_id={right_foot_body_id}")
    print(f"left_knee_joint_id={left_knee_joint_id}")
    print(f"right_knee_joint_id={right_knee_joint_id}")

    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    step_dt = get_step_dt(base_env)
    episode_length_s = get_episode_length_s(base_env)
    max_steps = int(math.ceil(episode_length_s / step_dt)) + 5

    fault_controller = FaultController(robot, joint_names, args_cli)

    obs = reset_env_get_obs(env)

    for ep in range(args_cli.episodes):
        fault_controller.restore()

        metrics = EpisodeMetricBuffer()
        fault_applied = False
        fault_applied_time_s = None
        fault_lock_angle = None

        obs = reset_env_get_obs(env)

        last_debug_t = -999.0
        timeout = False
        fall = False

        print(f"\nEpisode {ep + 1}/{args_cli.episodes} started.")

        for step in range(max_steps):
            t_s = step * step_dt

            fault_applied_now, lock_angle_now = fault_controller.apply_if_needed(t_s, fault_applied)
            if fault_applied_now and not fault_applied:
                fault_applied = True
                fault_applied_time_s = t_s
                fault_lock_angle = lock_angle_now

            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)

            metrics.update(
                t_s=t_s,
                robot=robot,
                left_foot_body_id=left_foot_body_id,
                right_foot_body_id=right_foot_body_id,
                left_knee_joint_id=left_knee_joint_id,
                right_knee_joint_id=right_knee_joint_id,
                foot_contact_height=args_cli.foot_contact_height,
            )

            done = unwrap_scalar_bool(dones)

            # Extra local fall heuristic in case the env does not terminate immediately.
            root_z = float(robot.data.root_pos_w[0, 2].detach().item())
            local_height_fall = root_z < float(args_cli.fall_height)

            if args_cli.debug and (t_s - last_debug_t) >= args_cli.debug_interval_s:
                x = float(robot.data.root_pos_w[0, 0].detach().item())
                y = float(robot.data.root_pos_w[0, 1].detach().item())
                vx = float(robot.data.root_lin_vel_b[0, 0].detach().item())
                vy = float(robot.data.root_lin_vel_b[0, 1].detach().item())
                print(
                    f"[DEBUG] ep={ep + 1} t={t_s:.2f}s "
                    f"x={x:.3f} y={y:.3f} z={root_z:.3f} "
                    f"vx_b={vx:.3f} vy_b={vy:.3f} "
                    f"done={done} fault={fault_applied}"
                )
                last_debug_t = t_s

            if done or local_height_fall:
                timeout = get_timeout_from_extras(extras, t_s, episode_length_s, done)
                fall = bool((done and not timeout) or local_height_fall)
                break

        row = metrics.finalize(
            task=args_cli.task,
            checkpoint=resume_path,
            episode=ep,
            fault_mode=args_cli.fault_mode,
            fault_joint=args_cli.fault_joint,
            fault_time_s=args_cli.fault_time_s,
            torque_scale=args_cli.torque_scale,
            fault_applied=fault_applied,
            fault_applied_time_s=fault_applied_time_s,
            fault_lock_angle=fault_lock_angle,
            timeout=timeout,
            fall=fall,
        )

        append_csv(args_cli.out_csv, row)

        print(
            f"[RESULT] ep={ep + 1} "
            f"fall={row['fall']} timeout={row['timeout']} "
            f"duration={row['episode_duration_s']:.3f}s "
            f"distance={row['forward_distance_m']} "
            f"mean_vx={row['mean_forward_vel_b_mps']} "
            f"min_z={row['min_root_height_m']}"
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
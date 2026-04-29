from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import json

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

#Using the G1_MINIMAL for training
from isaaclab_assets import G1_MINIMAL_CFG


@configclass
class G1BCPPOEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.0
    decimation = 2

    # PPO action is residual delta q added around BC target.
    action_space = 37

    # Observation:
    # q(37) + qd(37) + phase(1) + root angular velocity(3)
    # + projected gravity(3) + previous action(37)
    observation_space = 118
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # robot
    robot_cfg: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # files
    bc_policy_path: str = (
        r"C:\MAIN PROJECT CODE\Humanoid_Robot_Training_Under_DOF_Loss"
        r"\robomimic_WSL\bc_walking_policy_checkpoints\g1_bc_walk_V1\g1_model_epoch_81_best.pt"
    )

    bc_meta_json: str = (
        r"C:\MAIN PROJECT CODE\Humanoid_Robot_Training_Under_DOF_Loss"
        r"\data\BC_datasets\g1_walk_reference_bc_1024_regular.json"
    )

    # control for Unitree G1 environment motion and spawn height
    residual_scale: float = 0.25
    target_root_height: float = 0.70
    fall_height: float = 0.45
    gait_period_s: float = 4.25

    # ----------------REWARD WEIGHTS IMPORTANT TUNE-----------------------------
    #rew_pose: float = 1.5
    rew_pose: float = 0.75
    rew_vel: float = 0.10
    rew_bc: float = 0.25

    # Higher values here prioritize staying upright and not falling
    rew_upright: float = 3.0
    rew_height: float = 2.0
    rew_alive: float = 1.0

    # PENALTY
    penalty_action_rate: float = 0.02
    penalty_joint_vel: float = 0.001
    penalty_fall: float = 5.0


class G1BCPPOEnv(DirectRLEnv):
    cfg: G1BCPPOEnvCfg

    '''Loads BC metadata, action bounds, mapped walking .npz, loads torchscript BC policy and allocate runtime tensors'''
    def __init__(self, cfg: G1BCPPOEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Load metadata
        meta_path = Path(self.cfg.bc_meta_json)
        if not meta_path.exists():
            raise FileNotFoundError(f"BC metadata JSON not found: {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        self.joint_names_file = meta["joint_names"]

        self.action_low = torch.tensor(meta["action_low"], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(meta["action_high"], dtype=torch.float32, device=self.device)

        source_npz = Path(meta["source_mapped_npz"])
        if not source_npz.exists():
            raise FileNotFoundError(f"Mapped reference npz not found: {source_npz}")

        with np.load(source_npz, allow_pickle=True) as data:
            reference_q_np = np.asarray(data["joint_targets"], dtype=np.float32)

        self.reference_q = torch.tensor(reference_q_np, dtype=torch.float32, device=self.device)
        self.num_ref_frames = self.reference_q.shape[0]

        # Finite-difference reference velocity.
        dt_ref = 1.0 / float(meta.get("fps", 60.0))
        ref_qd_np = np.zeros_like(reference_q_np, dtype=np.float32)
        ref_qd_np[1:-1] = (reference_q_np[2:] - reference_q_np[:-2]) / (2.0 * dt_ref)
        ref_qd_np[0] = (reference_q_np[1] - reference_q_np[0]) / dt_ref
        ref_qd_np[-1] = (reference_q_np[-1] - reference_q_np[-2]) / dt_ref
        self.reference_qd = torch.tensor(ref_qd_np, dtype=torch.float32, device=self.device)

        # Load BC teacher policy as pre-trained foundation. PPO learns balance foundations around BC walking prior
        policy_path = Path(self.cfg.bc_policy_path)
        if not policy_path.exists():
            raise FileNotFoundError(f"BC TorchScript policy not found: {policy_path}")

        self.bc_policy = torch.jit.load(str(policy_path), map_location=self.device)
        self.bc_policy.eval()

        # Runtime buffers
        self.phase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, dtype=torch.float32, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        self.q_bc = torch.zeros_like(self.actions)
        self.q_target = torch.zeros_like(self.actions)

        # Episode logging buffers for TensorBoard. Debugging to measure training progress
        self._episode_sums = {
            "pose": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "vel": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "bc": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "upright": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "height": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "alive": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "action_rate_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "joint_vel_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "fall_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "total": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "episode_length": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "root_height": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "pose_error": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "bc_residual_l2": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
        }


    #Simulation bare bones isaaclab flat environment
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    '''Takes PPO actions in normalized [-1, 1] builds 75D BC observation and asks BC polcu for a normalized action then
    denormalizes it to q_bc and adds PPO algorithm residual'''
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)

        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel

        bc_obs = torch.cat([q, qd, self.phase.unsqueeze(-1)], dim=-1)

        with torch.no_grad():
            bc_action_norm = self.bc_policy(bc_obs)
            bc_action_norm = bc_action_norm.clamp(-1.0, 1.0)

        self.q_bc = 0.5 * (bc_action_norm + 1.0) * (self.action_high - self.action_low) + self.action_low

        residual = self.cfg.residual_scale * self.actions
        self.q_target = torch.clip(self.q_bc + residual, self.action_low, self.action_high)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.q_target)

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel

        root_ang_vel_b = self.robot.data.root_ang_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b

        obs = torch.cat(
            [
                q,
                qd,
                self.phase.unsqueeze(-1),
                root_ang_vel_b,
                projected_gravity_b,
                self.prev_actions,
            ],
            dim=-1,
        )

        return {"policy": obs}

    '''Key piece of code to tune to determine with pytorch the reward functions for PPO RL algorithm and what should be rewarded for walking policy.
    rewards robot for staying close to the reference and close to BC prior. Maintaining height and avoiding excessive action/joint velocity.
    Training behaviour is shaped here'''

    def _get_rewards(self) -> torch.Tensor:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel
        root_z = self.robot.data.root_pos_w[:, 2]

        ref_idx = torch.remainder((self.phase * self.num_ref_frames).long(), self.num_ref_frames)
        q_ref = self.reference_q[ref_idx]
        qd_ref = self.reference_qd[ref_idx]

        pose_error = torch.mean(torch.abs(q - q_ref), dim=-1)
        pose_reward = torch.exp(-4.0 * torch.mean((q - q_ref) ** 2, dim=-1))

        vel_error = torch.mean(torch.abs(qd - qd_ref), dim=-1)
        vel_reward = torch.exp(-0.25 * torch.mean((qd - qd_ref) ** 2, dim=-1))

        bc_residual_l2 = torch.mean((self.q_target - self.q_bc) ** 2, dim=-1)
        bc_reward = torch.exp(-4.0 * bc_residual_l2)

        projected_gravity = self.robot.data.projected_gravity_b
        upright_error = torch.sum(projected_gravity[:, :2] ** 2, dim=-1)
        upright_reward = torch.exp(-4.0 * upright_error)

        height_error = (root_z - self.cfg.target_root_height) ** 2
        height_reward = torch.exp(-20.0 * height_error)

        action_rate_penalty = torch.mean((self.actions - self.prev_actions) ** 2, dim=-1)
        joint_vel_penalty = torch.mean(qd**2, dim=-1)

        fallen = root_z < self.cfg.fall_height

        # All based of reward terms better to add G1BCPPOEnvCfg
        pose_term = self.cfg.rew_pose * pose_reward
        vel_term = self.cfg.rew_vel * vel_reward
        bc_term = self.cfg.rew_bc * bc_reward
        upright_term = self.cfg.rew_upright * upright_reward
        height_term = self.cfg.rew_height * height_reward
        alive_term = self.cfg.rew_alive * torch.ones_like(root_z)

        action_rate_term = -self.cfg.penalty_action_rate * action_rate_penalty
        joint_vel_term = -self.cfg.penalty_joint_vel * joint_vel_penalty
        fall_term = -self.cfg.penalty_fall * fallen.float()

        reward = (
            alive_term
            + pose_term
            + vel_term
            + bc_term
            + upright_term
            + height_term
            + action_rate_term
            + joint_vel_term
            + fall_term
        )

        # Accumulate episode statistics for TensorBoard logging.
        self._episode_sums["pose"] += pose_term
        self._episode_sums["vel"] += vel_term
        self._episode_sums["bc"] += bc_term
        self._episode_sums["upright"] += upright_term
        self._episode_sums["height"] += height_term
        self._episode_sums["alive"] += alive_term
        self._episode_sums["action_rate_penalty"] += action_rate_term
        self._episode_sums["joint_vel_penalty"] += joint_vel_term
        self._episode_sums["fall_penalty"] += fall_term
        self._episode_sums["total"] += reward
        self._episode_sums["episode_length"] += 1.0
        self._episode_sums["root_height"] += root_z
        self._episode_sums["pose_error"] += pose_error
        self._episode_sums["bc_residual_l2"] += bc_residual_l2

        self.prev_actions = self.actions.clone()

        # Advance phase at the RL control rate.
        control_dt = self.cfg.sim.dt * self.cfg.decimation
        self.phase = torch.remainder(self.phase + control_dt / self.cfg.gait_period_s, 1.0)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_z = self.robot.data.root_pos_w[:, 2]

        fallen = root_z < self.cfg.fall_height
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return fallen, time_out

    '''Reset during training if falls'''
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        

        if isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids.to(device=self.device)
        else:
            env_ids_t = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        num_reset = len(env_ids_t)

        #Tesnorboard logs record when environments resets after fall or timeout useful for understanding balance training
        if num_reset > 0 and hasattr(self, "_episode_sums"):
            extras = {}

            ep_len = torch.clamp(self._episode_sums["episode_length"][env_ids_t], min=1.0)

            for key, value in self._episode_sums.items():
                if key == "episode_length":
                    continue

                episodic_avg = torch.mean(value[env_ids_t] / ep_len)
                extras[f"Episode/{key}"] = episodic_avg.item()



            fallen_count = torch.count_nonzero(self.reset_terminated[env_ids_t]).item()
            timeout_count = torch.count_nonzero(self.reset_time_outs[env_ids_t]).item()

            extras["Episode/mean_length_steps"] = torch.mean(ep_len).item()
            extras["Episode/mean_length_seconds"] = torch.mean(ep_len * self.step_dt).item()
            extras["Episode_Termination/fallen"] = fallen_count
            extras["Episode_Termination/time_out"] = timeout_count
            extras["Episode_Termination/fallen_rate"] = fallen_count / max(num_reset, 1)
            extras["Episode_Termination/time_out_rate"] = timeout_count / max(num_reset, 1)

            self.extras["log"] = extras

            for key in self._episode_sums.keys():
                self._episode_sums[key][env_ids_t] = 0.0

        # Now call the DirectRLEnv reset
        super()._reset_idx(env_ids_t)



        # Start all envs from reference frame 0 first. Later, randomize this for robustness when learning
        joint_pos = self.reference_q[0].unsqueeze(0).repeat(num_reset, 1)
        joint_vel = torch.zeros_like(joint_pos)

        default_root_state = self.robot.data.default_root_state[env_ids_t].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids_t]
        default_root_state[:, 2] = self.cfg.target_root_height

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids_t)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_t)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_t)

        self.phase[env_ids_t] = 0.0
        self.actions[env_ids_t] = 0.0
        self.prev_actions[env_ids_t] = 0.0
        self.q_bc[env_ids_t] = joint_pos
        self.q_target[env_ids_t] = joint_pos
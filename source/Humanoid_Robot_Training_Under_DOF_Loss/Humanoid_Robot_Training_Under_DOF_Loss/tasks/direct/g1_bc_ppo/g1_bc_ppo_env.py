from __future__ import annotations

# Copyright (c) 2026, Mikolaj Wyrzykowski
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.utils.math import quat_apply_inverse

from isaaclab.sensors import ContactSensor, ContactSensorCfg

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
    # q(37) + qd(37) + phase(1)
    # + root angular velocity(3)
    # + projected gravity(3)
    # + root linear velocity(3)
    # + root height(1)
    # + previous action(37)
    observation_space = 122
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # robot
    robot_cfg: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Required for ContactSensor to work on G1 links
    robot_cfg.spawn.activate_contact_sensors = True

    # files, change these to match external project location of the base path
    bc_policy_path: str = (
        r"C:\MAIN PROJECT CODE\Humanoid_Robot_Training_Under_DOF_Loss"
        r"\bc_walking_policy_checkpoints\g1_bc_walk\g1_model_epoch_81_best.pt"
    )

    bc_meta_json: str = (
        r"C:\MAIN PROJECT CODE\Humanoid_Robot_Training_Under_DOF_Loss"
        r"\data\bc_dataset_demonstrations\g1_walk_reference_bc_1024_regular.json"
    )

    # ----------------REWARD WEIGHTS IMPORTANT TUNE-----------------------------

    # control for Unitree G1 environment motion and spawn height standard usually constant
    residual_scale: float = 0.15
    target_root_height: float = 0.70
    fall_height: float = 0.55
    gait_period_s: float = 4.25

    rew_pose: float = 0.12
    rew_vel: float = 0.01
    rew_bc: float = 0.02 # How much rewards being similar to BC prior

    # Higher values here prioritize staying upright and not falling. Typical for walking policy big penalty fall and big upright reward
    rew_upright: float = 4.5
    rew_height: float = 2.5
    rew_alive: float = 0.2
    rew_standing_height: float = 0.3

    penalty_low_height: float = 14.0
    penalty_knee_crouch: float = 0.14
    penalty_knee_collapse: float = 3.0

    # PENALTY RATES falling and moving joints out of predicted action
    penalty_action_rate: float = 0.025
    penalty_joint_vel: float = 0.001
    penalty_fall: float = 15.0

    # Lateral balance stability terms higher values reward more staying central
    penalty_lateral_vel: float = 2.4
    penalty_base_ang_vel: float = 0.35
    penalty_side_tilt: float = 2.4
    penalty_backward_vel: float = 4.0
    penalty_yaw_rate: float = 1.2

    min_good_root_height: float = 0.64
    standing_height_start: float = 0.60
    standing_height_full: float = 0.68
    knee_collapse_threshold: float = 0.90

    # Penalise one knee bending much more than the other during unstable stepping or knees far apart
    penalty_knee_asymmetry: float = 0.00

    #Walking velocity rewards and penalty
    target_forward_vel: float = 0.12 # m/s movement forward essentially
    rew_forward_vel: float = 0.16

    #Reward specifically lower body movement in a gait cycle matching BC prior
    rew_lower_body_gait: float = 0.12

    # Swing / trailing-foot recovery terms for stable gait
    rew_trailing_foot_recovery: float = 0.00
    rew_swing_foot_clearance: float = 0.00
    swing_clearance_target: float = 0.045
    target_swing_foot_forward_vel: float = 0.08
    penalty_foot_x_gap: float = 0.5
    max_foot_x_gap: float = 0.42

    # Phase-gated stepping terms terms
    rew_phase_swing_lift: float = 1.5
    rew_phase_forward_swing: float = 1.15
    rew_phase_single_support: float = 0.9
    penalty_wrong_phase_lift: float = 0.75
    phase_gate_power: float = 0.70

    '''------------Added contact airtime support rewards--------------------'''
    left_foot_contact_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/left_ankle_roll_link",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    right_foot_contact_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_ankle_roll_link",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    # Tuntable contact, airtime and weight distribution terms for leg support during step
    contact_force_threshold: float = 20.0
    rew_weight_shift: float = 0.5
    rew_single_support: float = 0.5
    rew_foot_airtime: float = 0.45
    penalty_foot_slip: float = 0.7

    min_air_time: float = 0.025
    target_air_time: float = 0.12
    max_contact_foot_speed: float = 0.12

    #Reward any foot lift and penalize standing still
    rew_any_foot_lift: float = 0.00
    rew_lift_unload: float = 0.5
    penalty_static_stand: float = 2.0

    #More configurations to force forward step swing behaviour
    min_counted_lift: float = 0.02
    rew_forward_swing_step: float = 0.6
    rew_sustained_swing_air: float = 1.0
    penalty_toe_tap: float = 3.5
    penalty_spin_step: float = 12.0
    
    #Penalities for missing the right phase alternating gait properly
    penalty_phase_missed_lift: float = 0.0
    penalty_phase_wrong_support_air: float = 0.25
    rew_phase_swing_air: float = 0.8

    #Tempt to allow the left leg to catch up reward wise for lifting in alternating gait
    rew_left_phase_lift_boost: float = 0.70
    rew_left_phase_forward_boost: float = 0.3
    rew_left_phase_right_support: float = 1.25
    rew_left_support_lift_combo: float = 0.8
    rew_left_step_touchdown: float = 1.5

    rew_right_phase_lift_boost: float = 4.0
    rew_right_phase_forward_boost: float = 2.0
    rew_right_support_lift_combo: float = 1.5
    rew_right_step_touchdown: float = 1.2

    #Terms to help alternating leg support, particularly stopping right leg from swinging forward and becoming more of a support leg
    penalty_left_phase_right_air: float = 2.0
    penalty_left_phase_left_heavy: float = 1.25
    penalty_right_re_lift_during_left_phase: float = 1.5
    penalty_short_right_stance_for_left: float = 1.25
    rew_right_stance_for_left: float = 1.8
    target_right_stance_time: float = 0.07
    

    # Force left foot to lift up right is holding down contact and weight, became overcomplicated so didnt use for discovering lift just to stabalize mirrored for right foot
    rew_left_unload_when_right_ready: float = 0.0
    rew_left_lift_when_right_ready: float = 2.0
    rew_left_up_vel_when_right_ready: float = 1.0
    rew_left_knee_flex_when_right_ready: float = 0.0
    rew_left_airborne_when_right_ready: float = 1.5
    rew_left_air_fwd_when_right_ready: float = 1.0

    rew_right_up_vel_when_left_ready: float = 3.0
    rew_right_airborne_when_left_ready: float = 1.5
    penalty_right_no_lift_when_left_ready: float = 7.0
    target_right_lift: float = 0.045
    rew_right_lift_discovery: float = 7.0
    rew_right_up_vel_discovery: float = 3.0

    # Mirrored support preparation for right-foot swing
    rew_left_stance_for_right: float = 1.5
    penalty_short_left_stance_for_right: float = 1.0
    target_left_stance_time: float = 0.06


    
    penalty_left_contact_when_right_ready: float = 2.8
    penalty_left_drag_during_left_swing: float = 2.0
    penalty_left_no_lift_when_right_ready: float = 7.0
    penalty_left_load_during_left_swing: float = 2.0
    penalty_right_drag_discovery: float = 0.75
    penalty_right_phase_missed_lift_soft: float = 1.4

    
    rew_phase_airtime_hold: float = 0.45
    rew_phase_air_forward: float = 1.4
    rew_forward_direction: float = 0.8
    rew_phase_high_lift: float = 1.2
    target_phase_lift: float = 0.055

    penalty_swing_lateral_vel: float = 1.3
    penalty_lateral_dominance: float = 9.0



class G1BCPPOEnv(DirectRLEnv):
    cfg: G1BCPPOEnvCfg

    '''Loads BC metadata, action bounds, mapped walking .npz, loads torchscript BC policy and allocate runtime tensors'''
    def __init__(self, cfg: G1BCPPOEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.robot_cfg.spawn.activate_contact_sensors = True
        super().__init__(cfg, render_mode, **kwargs)

        # Load metadata
        meta_path = Path(self.cfg.bc_meta_json)
        if not meta_path.exists():
            raise FileNotFoundError(f"BC metadata JSON not found: {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        self.joint_names_file = meta["joint_names"]

        self.left_knee_idx = self.joint_names_file.index("left_knee_joint")
        self.right_knee_idx = self.joint_names_file.index("right_knee_joint")

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

        self.left_air_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.right_air_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.prev_left_contact = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_right_contact = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        self.left_stance_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.right_stance_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.lower_body_joint_names = [
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

        self.lower_body_joint_ids = torch.tensor(
            [self.joint_names_file.index(name) for name in self.lower_body_joint_names],
            dtype=torch.long,
            device=self.device,
        )

        # Foot / ankle body ids for swing-foot recovery rewards.
        left_foot_ids, left_foot_names = self.robot.find_bodies(".*left_ankle_roll.*")
        right_foot_ids, right_foot_names = self.robot.find_bodies(".*right_ankle_roll.*")

        if len(left_foot_ids) == 0 or len(right_foot_ids) == 0:
            raise RuntimeError( "Could not find left/right ankle roll bodies." "Print self.robot.body_names and update the foot body name patterns.")

        self.left_foot_body_id = left_foot_ids[0]
        self.right_foot_body_id = right_foot_ids[0]

        print(f"[INFO] Left foot body: {left_foot_names[0]} id={self.left_foot_body_id}")
        print(f"[INFO] Right foot body: {right_foot_names[0]} id={self.right_foot_body_id}")



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
            "lateral_vel_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "base_ang_vel_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "side_tilt_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "lateral_velocity": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "base_ang_vel": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "low_height_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "knee_crouch_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "mean_knee_angle": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "knee_collapse_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "max_knee_angle": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_knee_angle": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_knee_angle": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "knee_asymmetry": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "knee_asymmetry_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "standing_height": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "forward_vel": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "forward_vel_reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "backward_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "yaw_rate_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "lower_body_gait": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "lower_body_error": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "trailing_foot_recovery": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "swing_foot_clearance": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "trailing_foot_lift": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "trailing_foot_fwd_vel": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "foot_x_gap": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "foot_x_gap_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_contact": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_contact": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_force_z": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_force_z": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "weight_shift": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "single_support": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "foot_airtime": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "foot_slip_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_air_time": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_air_time": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "raw_single_support": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "raw_foot_airtime": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "raw_weight_shift": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "raw_foot_slip": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "swing_unload": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "static_stand_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "dense_swing_air": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "any_foot_lift": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "lift_unload": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "proper_forward_swing": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "toe_tap_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "spin_step_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "sustained_swing_air": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "counted_trailing_lift": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "phase_swing_lift": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "phase_forward_swing": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "phase_single_support": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "wrong_phase_lift": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "phase_active_gate": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "phase_swing_air": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "phase_missed_lift": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "phase_wrong_support_air": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_phase_lift": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_phase_lift": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_phase_lift_dense": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_phase_lift_dense": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_phase_lift_boost": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_phase_forward_boost": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_phase_right_support": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_support_lift_combo": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_phase_lift_boost": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_phase_forward_boost": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_support_lift_combo": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_step_touchdown": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_phase_right_air": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_phase_left_heavy": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_stance_for_left": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_re_lift_during_left": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_stance_time": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_load_frac": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_stance_short_for_left": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_unload_when_right_ready": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_lift_when_right_ready": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_up_vel_when_right_ready": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_knee_flex_when_right_ready": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_contact_when_right_ready": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_airborne_when_right_ready": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_air_fwd_when_right_ready": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_drag_during_left_swing": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_no_lift_when_right_ready": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_load_during_left_swing": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_unload_discovery": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_drag_discovery": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_lift_discovery": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_up_vel_discovery": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "raw_left_up_vel_discovery": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "phase_airtime_hold": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "phase_air_forward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "swing_lateral_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "forward_direction": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "lateral_dominance_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "phase_high_lift": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_no_lift_when_left_ready": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_up_vel_when_left_ready": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_airborne_when_left_ready": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_lift_discovery": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_up_vel_discovery": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_drag_discovery": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "raw_right_up_vel_discovery": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_step_touchdown": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_stance_for_right": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_stance_time_for_right": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_load_frac_for_right": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "left_support_ready_for_right": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_lift_demand_gate": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_lift_discovery_gate": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "right_phase_missed_lift_soft": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            
        }

    #Simulation bare bones isaaclab flat environment
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=True)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        self.left_foot_contact_sensor = ContactSensor(self.cfg.left_foot_contact_cfg)
        self.right_foot_contact_sensor = ContactSensor(self.cfg.right_foot_contact_cfg)

        self.scene.sensors["left_foot_contact"] = self.left_foot_contact_sensor
        self.scene.sensors["right_foot_contact"] = self.right_foot_contact_sensor    

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

        #Comment back in when running normal but comment out when checking BC prior by itself 0 residual.
        #self.q_target = torch.clip(self.q_bc, self.action_low, self.action_high)
        residual = self.cfg.residual_scale * self.actions
        self.q_target = torch.clip(self.q_bc + residual, self.action_low, self.action_high)
        

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.q_target)

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel

        root_ang_vel_b = self.robot.data.root_ang_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_z = self.robot.data.root_pos_w[:, 2].unsqueeze(-1)

        obs = torch.cat(
            [
                q,
                qd,
                self.phase.unsqueeze(-1),
                root_ang_vel_b,
                projected_gravity_b,
                root_lin_vel_b,
                root_z,
                self.prev_actions,
            ],
            dim=-1,
        )

        #Sanity check right amount of observations in training
        if obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"Observation dimension mismatch. Built {obs.shape[-1]}, "
                f"expected {self.cfg.observation_space}."
            )

        return {"policy": obs}

    '''Key piece of code to tune to determine with pytorch the reward functions for PPO RL algorithm and what should be rewarded for walking policy.
    rewards robot for staying close to the reference and close to BC prior. Maintaining height and avoiding excessive action/joint velocity.
    Training behaviour is shaped here'''

    def _get_rewards(self) -> torch.Tensor:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel

        root_z = self.robot.data.root_pos_w[:, 2]
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        
        #forward velocity target
        #Expected heading "direction" punished by exceeding yaw rate too much away from this
        forward_vel = root_lin_vel_b[:, 0]
        yaw_rate = root_ang_vel_b[:, 2]
        heading_gate = torch.exp(-3.0 * yaw_rate ** 2)

        ref_idx = torch.remainder((self.phase * self.num_ref_frames).long(), self.num_ref_frames)
        q_ref = self.reference_q[ref_idx]
        qd_ref = self.reference_qd[ref_idx]

        # rewards for lower gait and gait terms
        lower_q = q[:, self.lower_body_joint_ids]
        lower_q_ref = q_ref[:, self.lower_body_joint_ids]

        lower_body_error = torch.mean((lower_q - lower_q_ref) ** 2, dim=-1)
        lower_body_gait_reward = torch.exp(-8.0 * lower_body_error)

        lower_body_gait_term = self.cfg.rew_lower_body_gait * lower_body_gait_reward

        pose_error = torch.mean(torch.abs(q - q_ref), dim=-1)
        pose_reward = torch.exp(-4.0 * torch.mean((q - q_ref) ** 2, dim=-1))
        vel_reward = torch.exp(-0.25 * torch.mean((qd - qd_ref) ** 2, dim=-1))

        bc_residual_l2 = torch.mean((self.q_target - self.q_bc) ** 2, dim=-1)
        bc_reward = torch.exp(-4.0 * bc_residual_l2)

        projected_gravity = self.robot.data.projected_gravity_b
        upright_error = torch.sum(projected_gravity[:, :2] ** 2, dim=-1)
        upright_reward = torch.exp(-4.0 * upright_error)

        height_error = (root_z - self.cfg.target_root_height) ** 2

        #Height reward was configured here to be more punishing
        height_reward = torch.exp(-80.0 * height_error)
        standing_height_reward = torch.clamp((root_z - self.cfg.standing_height_start) / (self.cfg.standing_height_full - self.cfg.standing_height_start), 0.0, 1.0,)

        # ----------------------------------------------------------------------------------------------------------
        # Trailing-foot recovery reward TOOK LONG TIME, DESIGNED TO HELP LIFT FEAT FOR WALKING GAIT PROPERLY
        # ----------------------------------------------------------------------------------------------------------
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w

        left_foot_pos_w = self.robot.data.body_pos_w[:, self.left_foot_body_id]
        right_foot_pos_w = self.robot.data.body_pos_w[:, self.right_foot_body_id]

        left_foot_vel_w = self.robot.data.body_lin_vel_w[:, self.left_foot_body_id]
        right_foot_vel_w = self.robot.data.body_lin_vel_w[:, self.right_foot_body_id]

        # Convert foot positions and velocities to root/body frame.
        left_foot_pos_b = quat_apply_inverse(root_quat_w, left_foot_pos_w - root_pos_w)
        right_foot_pos_b = quat_apply_inverse(root_quat_w, right_foot_pos_w - root_pos_w)

        left_foot_vel_b = quat_apply_inverse(root_quat_w, left_foot_vel_w - root_lin_vel_w)
        right_foot_vel_b = quat_apply_inverse(root_quat_w, right_foot_vel_w - root_lin_vel_w)

        # Foot with smaller body-frame x is the trailing foot.
        left_is_trailing = left_foot_pos_b[:, 0] < right_foot_pos_b[:, 0] # This is currently a reactive phase gait for walking since BC prior is weak and not clear
        foot_x_gap = torch.abs(left_foot_pos_b[:, 0] - right_foot_pos_b[:, 0])
        foot_x_gap_penalty = torch.relu(foot_x_gap - self.cfg.max_foot_x_gap) ** 2
        foot_x_gap_term = -self.cfg.penalty_foot_x_gap * foot_x_gap_penalty
        trailing_gap_gate = torch.clamp(foot_x_gap / 0.10, 0.0, 1.0)
        trailing_foot_fwd_vel = torch.where(left_is_trailing, left_foot_vel_b[:, 0], right_foot_vel_b[:, 0],)

        # Estimate swing lift relative to the lower foot.
        left_foot_z = left_foot_pos_w[:, 2]
        right_foot_z = right_foot_pos_w[:, 2]
        foot_z_min = torch.minimum(left_foot_z, right_foot_z)

        left_lift = torch.clamp(left_foot_z - foot_z_min, min=0.0)
        right_lift = torch.clamp(right_foot_z - foot_z_min, min=0.0)

        any_foot_lift = torch.maximum(left_lift, right_lift)
        any_foot_lift_reward = torch.clamp(any_foot_lift / self.cfg.swing_clearance_target, 0.0, 1.0,)
        counted_lift_reward = torch.clamp((any_foot_lift - self.cfg.min_counted_lift) / (self.cfg.swing_clearance_target - self.cfg.min_counted_lift + 1e-6),0.0, 1.0,)

        trailing_foot_z = torch.where(left_is_trailing, left_foot_z, right_foot_z,)
        trailing_foot_lift = trailing_foot_z - foot_z_min
        # trailing foot lift should be rewarded in other terms such as the proper swing forward with the trailing leg in walking gait
        counted_trailing_lift_reward = torch.clamp( (trailing_foot_lift - self.cfg.min_counted_lift) / (self.cfg.swing_clearance_target - self.cfg.min_counted_lift + 1e-6), 0.0, 1.0,)

        # Reward the trailing foot moving forward relative to the base.
        trailing_foot_recovery_reward = torch.clamp(trailing_foot_fwd_vel / self.cfg.target_swing_foot_forward_vel, 0.0, 1.0,)

        # Reward a trailing_foot lift to mimic gait walking, clearance gate rewards foot lift first then recovery
        swing_foot_clearance_reward = torch.clamp(trailing_foot_lift / self.cfg.swing_clearance_target, 0.0, 1.0,)
        #clearance_gate = 0.75 + 0.25 * trailing_foot_recovery_reward
        clearance_gate = 1.0

        base_stability_gate = torch.exp(-0.5 * torch.sum(root_ang_vel_b**2, dim=-1))
        height_gate = torch.clamp((root_z - 0.60) / (0.67 - 0.60), 0.0, 1.0) # Robot when moving has bent knees and showed root height between 0.63-0.66 usually

        #When walking and training the phase gait robot abused having the smallest height_gate tanking other penalities and rewards
        phase_motion_gate = torch.clamp((root_z - 0.56) / (0.66 - 0.56), 0.35, 1.0)


        # Only reward swing behaviour while reasonably upright/tall and feet are far apart from each-other in walking trail
        #swing_gate = upright_reward * base_stability_gate * height_gate * trailing_gap_gate
        swing_gate = upright_reward * height_gate * trailing_gap_gate
        trailing_foot_recovery_term = (self.cfg.rew_trailing_foot_recovery* trailing_foot_recovery_reward * swing_gate)
        swing_foot_clearance_term = (self.cfg.rew_swing_foot_clearance * swing_foot_clearance_reward * swing_gate)

        #-----------------------END EXPERIMENT TRAILING FOOT RECOVERY ----------------------------------------------


        #-------------------------CONTACT AWARE BEHAVIOUR TERMS-----------------------------------------------------

        left_force_w = self.left_foot_contact_sensor.data.net_forces_w[:, 0, :]
        right_force_w = self.right_foot_contact_sensor.data.net_forces_w[:, 0, :]

        left_force_z = torch.clamp(left_force_w[:, 2], min=0.0)
        right_force_z = torch.clamp(right_force_w[:, 2], min=0.0)

        left_contact = (left_force_z > self.cfg.contact_force_threshold).float()
        right_contact = (right_force_z > self.cfg.contact_force_threshold).float()

        # If left foot is trailing, right foot should be support.
        # If right foot is trailing, left foot should be support.
        support_force_z = torch.where(left_is_trailing, right_force_z, left_force_z)
        swing_force_z = torch.where(left_is_trailing, left_force_z, right_force_z)

        support_contact = torch.where(left_is_trailing, right_contact, left_contact)
        swing_contact = torch.where(left_is_trailing, left_contact, right_contact)
        swing_unload_reward = support_contact * (1.0 - swing_contact)

        swing_unload_term = (self.cfg.rew_single_support * swing_unload_reward * upright_reward * height_gate* trailing_gap_gate * heading_gate)


        # Reward unloading the swing/trailing foot and loading the support foot.
        weight_shift_reward = torch.clamp(
            (support_force_z - swing_force_z) / (support_force_z + swing_force_z + 1e-6),
            0.0,
            1.0,
        )

        weight_shift_term = (self.cfg.rew_weight_shift * weight_shift_reward * upright_reward * height_gate * trailing_gap_gate)

        # Reward true single support: support foot in contact, swing/trailing foot off contact.
        single_support_reward = support_contact * (1.0 - swing_contact)
        single_support_term = (self.cfg.rew_single_support * single_support_reward * (0.25 + 0.75 * swing_foot_clearance_reward) * upright_reward * height_gate * heading_gate)

        # Airtime timers for both feet.
        dt = self.step_dt

        # Reward term for holding feet in the air for longer after lifting foot
        left_new_air_time = self.left_air_time + (1.0 - left_contact) * dt
        right_new_air_time = self.right_air_time + (1.0 - right_contact) * dt

        left_touchdown = (left_contact > 0.5) & (self.prev_left_contact < 0.5)
        right_touchdown = (right_contact > 0.5) & (self.prev_right_contact < 0.5)

        left_airtime_bonus = torch.where(
            left_touchdown,
            torch.clamp(
                (self.left_air_time - self.cfg.min_air_time)

                / (self.cfg.target_air_time - self.cfg.min_air_time),
                0.0,
                1.0,
            ),
            torch.zeros_like(root_z),
        )

        right_airtime_bonus = torch.where(
            right_touchdown,
            torch.clamp(
                (self.right_air_time - self.cfg.min_air_time)
                / (self.cfg.target_air_time - self.cfg.min_air_time),
                0.0,
                1.0,
            ),
            torch.zeros_like(root_z),
        )

        foot_airtime_reward = left_airtime_bonus + right_airtime_bonus

        foot_airtime_term = (self.cfg.rew_foot_airtime * foot_airtime_reward * upright_reward * height_gate)

        # Reset airtime on contact, keep accumulating while airborne.
        self.left_air_time = torch.where(left_contact > 0.5, torch.zeros_like(root_z), left_new_air_time)
        self.right_air_time = torch.where(right_contact > 0.5, torch.zeros_like(root_z), right_new_air_time)

        self.prev_left_contact = left_contact.clone()
        self.prev_right_contact = right_contact.clone()

        # Penalise sliding/dragging while the foot is in contact.
        left_foot_speed_xy = torch.norm(left_foot_vel_b[:, :2], dim=-1)
        right_foot_speed_xy = torch.norm(right_foot_vel_b[:, :2], dim=-1)

        left_new_stance_time = torch.where(left_contact > 0.5, self.left_stance_time + dt, torch.zeros_like(root_z),)
        right_new_stance_time = torch.where(right_contact > 0.5, self.right_stance_time + dt, torch.zeros_like(root_z),)

        left_slip_penalty = left_contact * torch.relu(left_foot_speed_xy - self.cfg.max_contact_foot_speed) ** 2
        right_slip_penalty = right_contact * torch.relu(right_foot_speed_xy - self.cfg.max_contact_foot_speed) ** 2

        foot_slip_penalty = left_slip_penalty + right_slip_penalty
        foot_slip_term = -self.cfg.penalty_foot_slip * foot_slip_penalty


        #-----------------------------Finished contact-aware-------------------------------------------

        support_exists = torch.maximum(left_contact, right_contact)

        # OLD foot lift activity term any_foot_lift_term = (self.cfg.rew_any_foot_lift * any_foot_lift_reward * support_exists * upright_reward * height_gate)
        any_foot_lift_term = (self.cfg.rew_any_foot_lift * counted_trailing_lift_reward * support_exists * upright_reward * height_gate)

        action_rate_penalty = torch.mean((self.actions - self.prev_actions) ** 2, dim=-1)
        joint_vel_penalty = torch.mean(qd**2, dim=-1)

        # Reward positive forward motion up to target speed. Dont reward standing still, have to keep up motion gait
        forward_progress_reward = torch.clamp( forward_vel / self.cfg.target_forward_vel, 0.0, 1.0,)
        forward_vel_reward = forward_progress_reward

        forward_direction_reward = torch.clamp((forward_vel - torch.abs(root_lin_vel_b[:, 1])) / self.cfg.target_forward_vel, 0.0, 1.0,)
        forward_direction_term = (self.cfg.rew_forward_direction * forward_direction_reward * upright_reward * standing_height_reward)


        #Key balance gate metric introduced to help forward term reward with moving forward while standing upright
        balance_gate = upright_reward * standing_height_reward

        backward_penalty = torch.relu(-forward_vel) ** 2
        yaw_rate_penalty = yaw_rate ** 2

        backward_term = -self.cfg.penalty_backward_vel * backward_penalty
        yaw_rate_term = -self.cfg.penalty_yaw_rate * yaw_rate_penalty

        # Sideways drift is bad during survival training and during forward gait step
        lateral_vel_penalty = root_lin_vel_b[:, 1] ** 2
        lateral_dominance_penalty = torch.relu(torch.abs(root_lin_vel_b[:, 1]) - 0.65 * torch.relu(forward_vel)) ** 2
        lateral_dominance_term = (-self.cfg.penalty_lateral_dominance * lateral_dominance_penalty * upright_reward * height_gate)
        

        # Excessive base rotation usually appears before falling.
        base_ang_vel_penalty = torch.sum(root_ang_vel_b**2, dim=-1)

        # projected_gravity[:, 0] and [:, 1] represent tilt away from upright.
        side_tilt_penalty = torch.sum(projected_gravity[:, :2] ** 2, dim=-1)

        # Penalise crouched survival below a useful standing height.
        low_height_penalty = torch.relu(self.cfg.min_good_root_height - root_z)

        # Penalise excessive knee flexing. Penalize too much knee asymmetry that would be expected in a stable walk
        knee_angles = q[:, [self.left_knee_idx, self.right_knee_idx]]
        mean_knee_angle = torch.mean(knee_angles, dim=-1)
        max_knee_angle = torch.max(knee_angles, dim=-1).values

        knee_asymmetry = torch.abs(knee_angles[:, 0] - knee_angles[:, 1])
        knee_asymmetry_penalty = knee_asymmetry ** 2
        knee_asymmetry_term = -self.cfg.penalty_knee_asymmetry * knee_asymmetry_penalty

        # Only penalise crouching beyond a moderate knee bend. This got bumped up as my robot learnt to survive taller and this was too low
        knee_crouch_penalty = torch.mean(torch.relu(knee_angles - 0.60) ** 2, dim=-1)
        # Only punish severe knee collapse, not normal walking knee bend.
        knee_collapse_penalty = torch.mean(torch.relu(knee_angles - self.cfg.knee_collapse_threshold) ** 2, dim=-1,)
        knee_collapse_term = -self.cfg.penalty_knee_collapse * knee_collapse_penalty

        not_crouched_gate = torch.clamp((0.82 - mean_knee_angle) / 0.12, 0.0, 1.0)


        # Reward actual airtime lift only not just airtime and then touchdown
        dense_swing_air_reward = (support_contact * (1.0 - swing_contact) * swing_foot_clearance_reward)
        dense_swing_air_term = (0.25 * dense_swing_air_reward * upright_reward * height_gate * heading_gate)

        lift_unload_reward = (swing_foot_clearance_reward * weight_shift_reward * support_contact)
        lift_unload_term = (self.cfg.rew_lift_unload * lift_unload_reward * upright_reward * height_gate * heading_gate)

        #Key term for moving forward with an unloaded foot and support foot
        forward_swing_reward = torch.clamp(trailing_foot_fwd_vel / self.cfg.target_swing_foot_forward_vel, 0.0, 1.0,)
        proper_forward_swing_reward = (single_support_reward * counted_trailing_lift_reward * forward_swing_reward * weight_shift_reward)
        proper_forward_swing_term = (self.cfg.rew_forward_swing_step * proper_forward_swing_reward * upright_reward * height_gate * heading_gate)

        # Noticed in PPO_WALK_004 problems with rewarding short lifts and placing foot back down want to punish this behaviour to encourage stepping
        toe_tap_penalty = (single_support_reward * (1.0 - counted_trailing_lift_reward) * torch.clamp(torch.abs(trailing_foot_fwd_vel) / 0.05, 0.0, 1.0))
        toe_tap_term = (-self.cfg.penalty_toe_tap * toe_tap_penalty * upright_reward * height_gate)

        # Noticed in PPO_WALK_004 any foot activity related to walking gait sometimes exceeded yaw rate and started spinning or twitching with the movement
        spin_step_penalty = single_support_reward * torch.relu(torch.abs(yaw_rate) - 0.35) ** 2
        spin_step_term = -self.cfg.penalty_spin_step * spin_step_penalty

        # --------------------------------------------------------------------------------
        # Soft phase gait gate:
        # phase 0.25 = right swing peak
        # phase 0.75 = left swing peak
        # phase 0.00 / 0.50 = transition
        # --------------------------------------------------------------------------------

        phase_angle = 2.0 * torch.pi * self.phase

        right_phase_swing_gate = torch.clamp(torch.sin(phase_angle), 0.0, 1.0)
        left_phase_swing_gate = torch.clamp(-torch.sin(phase_angle), 0.0, 1.0)

        # Broaden the timing window slightly so PPO is not punished too sharply.
        right_phase_swing_gate = right_phase_swing_gate ** self.cfg.phase_gate_power
        left_phase_swing_gate = left_phase_swing_gate ** self.cfg.phase_gate_power

        phase_left_should_swing = left_phase_swing_gate >= right_phase_swing_gate
        phase_swing_gate = torch.maximum(left_phase_swing_gate, right_phase_swing_gate)

        # Ignore very small gate values around double-support transitions.
        phase_active_gate = torch.clamp((phase_swing_gate - 0.15) / 0.85, 0.0, 1.0)

        phase_swing_lift = torch.where(phase_left_should_swing, left_lift, right_lift)
        phase_wrong_lift = torch.where(phase_left_should_swing, right_lift, left_lift)

        phase_swing_fwd_vel = torch.where( phase_left_should_swing, left_foot_vel_b[:, 0], right_foot_vel_b[:, 0],)
        phase_swing_contact = torch.where(phase_left_should_swing, left_contact, right_contact,)

        phase_support_contact = torch.where(phase_left_should_swing, right_contact, left_contact,)
        phase_support_gate = 0.25 + 0.75 * phase_support_contact

        phase_swing_force_z = torch.where(phase_left_should_swing, left_force_z, right_force_z,)
        phase_support_force_z = torch.where(phase_left_should_swing, right_force_z, left_force_z,)

        # Dense lift reward that gives learning signal from 0 cm upward.
        phase_lift_dense_reward = torch.clamp(phase_swing_lift / self.cfg.swing_clearance_target, 0.0, 1.0,)
        # Counted lift reward only rewards clear lift after small threshold.
        phase_lift_counted_reward = torch.clamp((phase_swing_lift - self.cfg.min_counted_lift) / (self.cfg.swing_clearance_target - self.cfg.min_counted_lift + 1e-6), 0.0, 1.0,)

        # Blend both dense reward and counted reward to help discovery for lifting feet clearance in walking gait
        phase_lift_reward = 0.60 * phase_lift_dense_reward + 0.40 * phase_lift_counted_reward
        #phase_lift_reward = torch.clamp((phase_swing_lift - self.cfg.min_counted_lift) / (self.cfg.swing_clearance_target - self.cfg.min_counted_lift + 1e-6), 0.0, 1.0,)
        phase_wrong_lift_reward = torch.clamp((phase_wrong_lift - self.cfg.min_counted_lift) / (self.cfg.swing_clearance_target - self.cfg.min_counted_lift + 1e-6), 0.0, 1.0,)
        phase_high_lift_reward = torch.clamp((phase_swing_lift - 0.015) / (self.cfg.target_phase_lift - 0.015 + 1e-6), 0.0, 1.0,)

        phase_high_lift_term = (
        self.cfg.rew_phase_high_lift
        * phase_high_lift_reward
        * phase_active_gate
        * phase_support_gate
        * upright_reward
        * height_gate
        * heading_gate
        )
        
        phase_forward_swing_reward = torch.clamp(phase_swing_fwd_vel / self.cfg.target_swing_foot_forward_vel, 0.0, 1.0,)
        phase_single_support_reward = phase_support_contact * (1.0 - phase_swing_contact)

        phase_weight_shift_reward = torch.clamp((phase_support_force_z - phase_swing_force_z) / (phase_support_force_z + phase_swing_force_z + 1e-6), 0.0, 1.0,)

        '''Logging to determine which leg is being ignored more in phase gait reward'''
        left_phase_lift_reward = torch.clamp((left_lift - self.cfg.min_counted_lift) / (self.cfg.swing_clearance_target - self.cfg.min_counted_lift + 1e-6), 0.0, 1.0,)
        right_phase_lift_reward = torch.clamp((right_lift - self.cfg.min_counted_lift) / (self.cfg.swing_clearance_target - self.cfg.min_counted_lift + 1e-6), 0.0, 1.0,)
        left_phase_lift_dense_reward = torch.clamp(left_lift / self.cfg.swing_clearance_target, 0.0, 1.0)
        right_phase_lift_dense_reward = torch.clamp(right_lift / self.cfg.swing_clearance_target, 0.0, 1.0)

        right_phase_missed_lift_soft_penalty = (
            right_phase_swing_gate
            * phase_active_gate
            * torch.clamp((self.cfg.target_right_lift - right_lift) / self.cfg.target_right_lift, 0.0, 1.0)
            * (0.35 + 0.65 * left_contact)
            * upright_reward
            * height_gate
            * heading_gate
        )

        right_phase_missed_lift_soft_term = (-self.cfg.penalty_right_phase_missed_lift_soft * right_phase_missed_lift_soft_penalty)



        left_phase_active = left_phase_swing_gate * phase_active_gate
        right_phase_active = right_phase_swing_gate * phase_active_gate

        gait_lift_gate = torch.clamp((phase_lift_dense_reward - 0.15) / 0.35, 0.0, 1.0)
        forward_vel_term = self.cfg.rew_forward_vel * forward_vel_reward * balance_gate * (0.25 + 0.75 * gait_lift_gate)

        force_sum_for_right = left_force_z + right_force_z + 1e-6
        right_load_frac_for_forward = right_force_z / force_sum_for_right
        left_load_frac_for_forward = left_force_z / force_sum_for_right


        right_phase_forward_reward = torch.clamp(right_foot_vel_b[:, 0] / self.cfg.target_swing_foot_forward_vel, 0.0, 1.0,)
        right_unload_reward = torch.clamp((0.70 - right_load_frac_for_forward) / 0.40, 0.0, 1.0)
        left_support_soft_for_right_forward = torch.clamp((left_load_frac_for_forward - 0.20) / 0.35, 0.0, 1.0)
        left_support_for_right_reward = (left_contact * torch.clamp((left_load_frac_for_forward - 0.15) / 0.35, 0.0, 1.0))
        left_stance_time_soft = torch.clamp(left_new_stance_time / 0.06, 0.0, 1.0)
        left_stance_time_target = torch.clamp(left_new_stance_time / self.cfg.target_left_stance_time, 0.0, 1.0)
        left_load_soft = torch.clamp((left_load_frac_for_forward - 0.10) / 0.30, 0.0, 1.0)
        left_plant_still_reward = torch.clamp((0.20 - left_foot_speed_xy) / 0.20, 0.0, 1.0)
        left_load_target = torch.clamp((left_load_frac_for_forward - 0.30) / 0.20, 0.0, 1.0)
        left_support_bridge_reward = left_contact * left_stance_time_soft * left_load_soft
        left_support_quality_reward = left_contact * left_stance_time_target * left_load_target * left_plant_still_reward
        left_support_for_right_soft = (left_contact * torch.clamp((left_load_frac_for_forward - 0.20) / 0.35, 0.0, 1.0))
        left_support_ready_for_right = (left_contact * left_stance_time_soft * left_load_soft * (0.50 + 0.50 * left_plant_still_reward))
        right_up_vel_discovery_reward = torch.clamp(right_foot_vel_w[:, 2] / 0.10, 0.0, 1.0)
        left_support_soft_gate_for_right = 0.5 + 0.5 * left_support_ready_for_right


        left_stance_for_right_term = (
        self.cfg.rew_left_stance_for_right
        * (0.65 * left_support_bridge_reward + 0.35 * left_support_quality_reward)
        * right_phase_swing_gate
        * phase_active_gate
        * upright_reward
        * height_gate
        * heading_gate
        )


        right_lift_discovery_gate = (
            right_phase_swing_gate
            * phase_active_gate
            * left_support_soft_gate_for_right
            * upright_reward
            * height_gate
            * heading_gate
        )

        right_lift_discovery_term = (
            self.cfg.rew_right_lift_discovery
            * right_lift_discovery_gate
            * right_phase_lift_dense_reward
            * (0.35 + 0.65 * right_phase_forward_reward)
        )

        right_up_vel_discovery_term = (
            self.cfg.rew_right_up_vel_discovery
            * right_phase_swing_gate
            * phase_active_gate
            * left_support_soft_gate_for_right
            * right_up_vel_discovery_reward
            * upright_reward
            * height_gate
            * heading_gate
        )

        right_drag_discovery_penalty = (
        right_phase_swing_gate
        * phase_active_gate
        * right_contact
        * torch.clamp(right_foot_vel_b[:, 0] / self.cfg.target_swing_foot_forward_vel, 0.0, 1.0)
        * (1.0 - right_phase_lift_dense_reward)
        * (0.25 + 0.75 * left_support_ready_for_right)
        * upright_reward
        * phase_motion_gate
        )

        right_step_place_reward = torch.clamp((right_foot_pos_b[:, 0] - left_foot_pos_b[:, 0] + 0.02) / 0.10, 0.0, 1.0,)
        right_step_touchdown_term = (
        self.cfg.rew_right_step_touchdown
        * right_touchdown.float()
        * right_airtime_bonus
        * right_step_place_reward
        * (0.35 + 0.65 * left_contact)
        * upright_reward
        * height_gate
        * heading_gate
        )



        right_drag_discovery_term = (-self.cfg.penalty_right_drag_discovery* right_drag_discovery_penalty)

        right_phase_lift_boost_term = (
        self.cfg.rew_right_phase_lift_boost
        * right_phase_lift_dense_reward
        * right_phase_swing_gate
        * phase_active_gate
        * (0.25 + 0.75 * left_contact)
        * upright_reward
        * height_gate
        * heading_gate
        )

        right_phase_forward_boost_term = (
        self.cfg.rew_right_phase_forward_boost
        * right_phase_lift_dense_reward
        * right_phase_forward_reward
        * right_unload_reward
        * right_phase_swing_gate
        * phase_active_gate
        * (0.35 + 0.65 * left_support_soft_for_right_forward)
        * (0.35 + 0.65 * left_contact)
        * upright_reward
        * height_gate
        * heading_gate
        )

        right_support_lift_combo_term = (
        self.cfg.rew_right_support_lift_combo
        * left_support_for_right_reward
        * right_phase_lift_dense_reward
        * right_phase_swing_gate
        * phase_active_gate
        * upright_reward
        * height_gate
        * heading_gate
        )

        right_lift_demand_gate = (
        right_phase_swing_gate
        * phase_active_gate
        * left_support_ready_for_right
        * upright_reward
        * height_gate
        * heading_gate
        )

        right_up_vel_reward = torch.clamp(right_foot_vel_b[:, 2] / 0.10, 0.0, 1.0)
        right_no_lift_penalty = (right_lift_demand_gate * torch.clamp( (self.cfg.target_right_lift - right_lift) / self.cfg.target_right_lift, 0.0, 1.0,))
        right_no_lift_term = (-self.cfg.penalty_right_no_lift_when_left_ready * right_no_lift_penalty)
        right_up_vel_when_left_ready_term = (self.cfg.rew_right_up_vel_when_left_ready * right_lift_demand_gate * right_up_vel_reward)
        right_airborne_when_left_ready_term = (self.cfg.rew_right_airborne_when_left_ready * right_lift_demand_gate * (1.0 - right_contact) * (0.25 + 0.75 * right_phase_lift_dense_reward))



        phase_swing_lift_term = ( 
        self.cfg.rew_phase_swing_lift 
        * phase_lift_reward 
        * phase_active_gate 
        * phase_support_gate
        * upright_reward 
        * phase_motion_gate 
        * heading_gate
        )

        phase_forward_swing_term = (
        self.cfg.rew_phase_forward_swing 
        * phase_lift_reward 
        * phase_forward_swing_reward 
        * (0.25 + 0.75 * phase_single_support_reward)
        * (0.25 + 0.75 *phase_weight_shift_reward)
        * phase_active_gate
        * upright_reward
        * phase_motion_gate
        * heading_gate
        )

        phase_single_support_term = (
        self.cfg.rew_phase_single_support
        * phase_single_support_reward
        * phase_active_gate
        * upright_reward
        * phase_motion_gate
        * heading_gate
        )

        wrong_phase_lift_term = (
        -self.cfg.penalty_wrong_phase_lift
        * phase_wrong_lift_reward
        * phase_active_gate
        * upright_reward
        * phase_motion_gate
        )

        phase_swing_air_time = torch.where(phase_left_should_swing, left_new_air_time, right_new_air_time,)
        phase_sustained_air_reward = torch.clamp((phase_swing_air_time - self.cfg.min_air_time) / (self.cfg.target_air_time - self.cfg.min_air_time + 1e-6), 0.0, 1.0,)

        phase_swing_lateral_vel = torch.where(phase_left_should_swing, left_foot_vel_b[:, 1], right_foot_vel_b[:, 1],)

        # Dense reward for keeping the correct swing foot airborne a little longer.
        # Starts rewarding after about 0.03s, saturates around 0.16s.
        phase_airtime_hold_reward = torch.clamp( (phase_swing_air_time - 0.005) / (0.09 - 0.005 + 1e-6), 0.0, 1.0,)

        phase_airtime_hold_term = (
        self.cfg.rew_phase_airtime_hold
        * phase_airtime_hold_reward
        * phase_lift_reward
        * phase_active_gate
        * upright_reward
        * height_gate
        * heading_gate
        )

        # Reward airborne swing only when it is also moving forward.
        phase_air_forward_term = (
        self.cfg.rew_phase_air_forward
        * phase_airtime_hold_reward
        * phase_lift_reward
        * phase_forward_swing_reward
        * phase_active_gate
        * upright_reward
        * height_gate
        * heading_gate
        )

        # Penalise sideways swing motion while the foot is lifted.
        swing_lateral_penalty = (
        phase_swing_lateral_vel ** 2
        * phase_lift_reward
        * phase_active_gate
        * upright_reward
        * height_gate
        )

        swing_lateral_term = -self.cfg.penalty_swing_lateral_vel * swing_lateral_penalty


        sustained_swing_air_term = (self.cfg.rew_sustained_swing_air
        * phase_sustained_air_reward
        * phase_lift_reward
        * phase_single_support_reward
        * phase_active_gate
        * upright_reward
        * height_gate
        * heading_gate
        )

        ''' Added some missing the alternating phase gait cycle penalities for not alternating the opposite leg and placing it down'''
        phase_swing_air_reward = torch.where(phase_left_should_swing, 1.0 - left_contact, 1.0 - right_contact,)

        phase_support_air_wrong = torch.where(phase_left_should_swing, 1.0 - right_contact, 1.0 - left_contact,)

        phase_missed_lift_penalty = (1.0 - phase_lift_reward) * phase_active_gate * phase_support_gate

        phase_swing_air_term = (
        self.cfg.rew_phase_swing_air
        * phase_swing_air_reward
        * phase_active_gate
        * phase_support_gate
        * upright_reward
        * phase_motion_gate
        * heading_gate
        )

        phase_missed_lift_term = (
        -self.cfg.penalty_phase_missed_lift
        * phase_missed_lift_penalty
        * upright_reward
        * phase_motion_gate
        )

        phase_wrong_support_air_term = (
        -self.cfg.penalty_phase_wrong_support_air
        * phase_support_air_wrong
        * phase_active_gate
        * upright_reward
        * phase_motion_gate
        )


        # Temporary left leg lift boost reward term
        left_phase_lift_dense_reward = torch.clamp(left_lift / self.cfg.swing_clearance_target, 0.0, 1.0)
        right_phase_lift_dense_reward = torch.clamp(right_lift / self.cfg.swing_clearance_target, 0.0, 1.0)

        left_phase_forward_reward = torch.clamp(left_foot_vel_b[:, 0] / self.cfg.target_swing_foot_forward_vel, 0.0, 1.0)

        left_phase_support_contact = right_contact
        left_unload_reward = torch.clamp((right_force_z - left_force_z) / (right_force_z + left_force_z + 1e-6), 0.0, 1.0)

        left_phase_lift_boost_term = (
        self.cfg.rew_left_phase_lift_boost
        * left_phase_lift_dense_reward
        * left_phase_swing_gate
        * phase_active_gate
        * (0.25 + 0.75 * right_contact)
        * upright_reward
        * phase_motion_gate
        * heading_gate
        )

        left_phase_forward_boost_term = (
        self.cfg.rew_left_phase_forward_boost
        * left_phase_lift_dense_reward
        * left_phase_forward_reward
        * left_unload_reward
        * left_phase_swing_gate
        * phase_active_gate
        * left_phase_support_contact
        * upright_reward
        * phase_motion_gate
        * heading_gate
        )

       # Soft right-support signal for teaching left swing.
        force_sum = right_force_z + left_force_z + 1e-6
        right_load_frac = right_force_z / force_sum
        left_load_frac = left_force_z / force_sum

        # Becomes non-zero before the right foot fully dominates support.
        right_support_for_left_reward = (right_contact * torch.clamp((right_load_frac - 0.15) / 0.35, 0.0, 1.0))

        # Mild penalty if, during left-swing phase, the right foot is not available as support.
        left_phase_right_air_penalty = (
        (1.0 - right_contact)
        * left_phase_swing_gate
        * phase_active_gate
        * upright_reward
        * phase_motion_gate
        * heading_gate
        )

        # Mild penalty if the robot keeps too much load on the left foot during left-swing phase.
        left_phase_left_heavy_penalty = (
        torch.relu(left_load_frac - 0.75) ** 2
        * left_phase_swing_gate
        * phase_active_gate
        * upright_reward
        * phase_motion_gate
        * heading_gate
        )

        left_phase_right_air_term = -self.cfg.penalty_left_phase_right_air * left_phase_right_air_penalty
        left_phase_left_heavy_term = -self.cfg.penalty_left_phase_left_heavy * left_phase_left_heavy_penalty

        # Encourage shifting weight to right leg but lots of reward happens when left foot lifts.
        left_phase_right_support_term = (
        self.cfg.rew_left_phase_right_support
        * right_support_for_left_reward
        * (0.25 + 0.75 * left_phase_lift_dense_reward)
        * left_phase_swing_gate
        * phase_active_gate
        * upright_reward
        * phase_motion_gate
        * heading_gate
        )

        # Stronger combined reward term. right foot supports while left foot actually lifts.
        left_support_lift_combo_term = (
        self.cfg.rew_left_support_lift_combo
        * right_support_for_left_reward
        * left_phase_lift_dense_reward
        * left_phase_swing_gate
        * phase_active_gate
        * upright_reward
        * phase_motion_gate
        * heading_gate
        )

        # Reward left foot touchdown after being airborne, preferably landing forward of the right foot.
        left_step_place_reward = torch.clamp((left_foot_pos_b[:, 0] - right_foot_pos_b[:, 0] + 0.02) / 0.10, 0.0, 1.0,)

        left_step_touchdown_term = (
        self.cfg.rew_left_step_touchdown
        * left_touchdown.float()
        * left_airtime_bonus
        * left_step_place_reward
        * upright_reward
        * phase_motion_gate
        * heading_gate
        )


        force_sum = left_force_z + right_force_z + 1e-6
        right_load_frac = right_force_z / force_sum
        left_load_frac = left_force_z / force_sum

        # During left-swing phase, right foot should be the support foot.
        left_swing_intent_gate = (
        left_phase_swing_gate
        * phase_active_gate
        * upright_reward
        * phase_motion_gate
        * heading_gate
        )

        # Reward right foot becoming a real support foot during left-swing phase.


        right_stance_time_soft = torch.clamp(right_new_stance_time / 0.06, 0.0, 1.0)
        right_stance_time_target = torch.clamp(right_new_stance_time / self.cfg.target_right_stance_time, 0.0, 1.0,)

        right_load_soft = torch.clamp((right_load_frac - 0.10) / 0.30, 0.0, 1.0)
        right_load_target = torch.clamp((right_load_frac - 0.30) / 0.20, 0.0, 1.0)

        right_plant_still_reward = torch.clamp((0.20 - right_foot_speed_xy) / 0.20, 0.0, 1.0,)

        right_support_bridge_reward = (right_contact * right_stance_time_soft * right_load_soft)
        right_support_quality_reward = (right_contact * right_stance_time_target * right_load_target * right_plant_still_reward)

        right_stance_for_left_term = (self.cfg.rew_right_stance_for_left * (0.65 * right_support_bridge_reward + 0.35 * right_support_quality_reward) * left_swing_intent_gate)

        # Right support is ready for a left-leg swing.
        # This gate is only high when the phase wants left swing AND the right foot is planted/load-bearing.
        right_ready_for_left = (
        right_contact
        * right_stance_time_soft
        * right_load_soft
        * right_plant_still_reward
        * left_swing_intent_gate
        )
        #reward the physical preparation for lift.
        left_up_vel_reward = torch.clamp(left_foot_vel_b[:, 2] / 0.12, 0.0, 1.0)

        left_knee_flex_reward = torch.clamp((q[:, self.left_knee_idx] - 0.55) / (0.75 - 0.55), 0.0, 1.0,)

        left_unload_gate = torch.clamp((0.65 - left_load_frac) / 0.35, 0.0, 1.0)

        left_knee_flex_when_right_ready_term = (self.cfg.rew_left_knee_flex_when_right_ready * right_ready_for_left * left_knee_flex_reward * (1.0 - left_phase_lift_dense_reward))

        left_not_trying_to_lift = 1.0 - left_up_vel_reward


        left_airborne_reward = 1.0 - left_contact
        left_forward_motion_reward = torch.clamp(left_foot_vel_b[:, 0] / self.cfg.target_swing_foot_forward_vel, 0.0, 1.0,)
        left_drag_forward_amount = torch.clamp(left_foot_vel_b[:, 0] / self.cfg.target_swing_foot_forward_vel, 0.0, 1.0,)

        left_drag_during_left_swing_penalty = (
            left_contact
            * left_drag_forward_amount
            * (1.0 - left_phase_lift_dense_reward)
            * right_contact
            * right_load_soft
            * left_swing_intent_gate
            )
        
        right_support_ready_for_left = (right_contact * right_stance_time_soft * right_load_soft * right_plant_still_reward)

        left_lift_demand_gate = (
            left_phase_swing_gate
            * phase_active_gate
            * right_support_ready_for_left
            * upright_reward
            * phase_motion_gate
            * heading_gate
            )

        left_required_lift = 0.045

        left_drag_during_left_swing_term = (-self.cfg.penalty_left_drag_during_left_swing * left_drag_during_left_swing_penalty)
        left_no_lift_when_right_ready_penalty = (left_lift_demand_gate * torch.clamp((left_required_lift - left_lift) / left_required_lift, 0.0, 1.0))

        left_no_lift_when_right_ready_term = (-self.cfg.penalty_left_no_lift_when_right_ready * left_no_lift_when_right_ready_penalty)
        left_up_vel_when_right_ready_term = (self.cfg.rew_left_up_vel_when_right_ready * left_lift_demand_gate * left_up_vel_reward * not_crouched_gate)

        # Reward unloading the left foot once the right foot is ready.
        # If left_load_frac is high, the left foot is still carrying too much weight.
        left_unload_when_right_ready_reward = torch.clamp((0.70 - left_load_frac) / 0.30, 0.0, 1.0,)
        left_unload_when_right_ready_term = (self.cfg.rew_left_unload_when_right_ready * right_ready_for_left * left_unload_when_right_ready_reward)

        # Reward actual left foot lift only when the right foot is ready.
        left_lift_when_right_ready_term = (self.cfg.rew_left_lift_when_right_ready *left_lift_demand_gate * left_phase_lift_dense_reward)

        left_contact_when_right_ready_term = (-self.cfg.penalty_left_contact_when_right_ready * left_lift_demand_gate * left_contact * (1.0 - left_phase_lift_dense_reward) * left_not_trying_to_lift)

        left_airborne_when_right_ready_term = (self.cfg.rew_left_airborne_when_right_ready * left_lift_demand_gate * (1.0 - left_contact) * (0.25 + 0.75 * left_phase_lift_dense_reward) * not_crouched_gate)
        #Penalise right foot being unavailable during the left-swing phase.
        # This remains active even before the left foot lifts much.
        right_re_lift_during_left_penalty = ((1.0 - right_contact) * (0.50 + 0.50 * (1.0 - left_phase_lift_dense_reward)) * left_swing_intent_gate)

        right_re_lift_during_left_term = (-self.cfg.penalty_right_re_lift_during_left_phase * right_re_lift_during_left_penalty)

        left_air_fwd_when_right_ready_term = (self.cfg.rew_left_air_fwd_when_right_ready * left_lift_demand_gate  * (1.0 - left_contact) * left_phase_lift_dense_reward * left_forward_motion_reward)

        # This specifically discourages the stomp/tap pattern.
        right_stance_short_penalty = (right_contact * (1.0 - right_stance_time_target) * left_swing_intent_gate)
        right_stance_short_for_left_term = (-self.cfg.penalty_short_right_stance_for_left * right_stance_short_penalty)

        left_load_during_left_swing_penalty = (torch.relu(left_load_frac - 0.45) ** 2 * left_swing_intent_gate * right_contact)

        left_load_during_left_swing_term = (-self.cfg.penalty_left_load_during_left_swing * left_load_during_left_swing_penalty)
        left_unload_discovery_reward = torch.clamp((0.70 - left_load_frac) / 0.35, 0.0, 1.0)

        # I cant get this left leg to lift in the gait so I want PPO to discover lifting more generally. Simplifying to this term
        left_lift_discovery_term = (
            2.5
            * left_phase_swing_gate
            * phase_active_gate
            * right_contact
            * (0.25 + 0.75 * left_unload_discovery_reward)
            * left_phase_lift_dense_reward
            * (0.50 + 0.50 * left_phase_forward_reward)
            * upright_reward
            * height_gate
            * heading_gate
        )

        left_unload_discovery_term = (
            1.8
            * left_phase_swing_gate
            * phase_active_gate
            * right_contact
            * left_unload_discovery_reward
            * upright_reward
            * height_gate
            * heading_gate
        )

        # Punish weird drag behaviour
        left_drag_discovery_penalty = (
            left_phase_swing_gate
            * phase_active_gate
            * left_contact
            * torch.clamp(left_foot_vel_b[:, 0] / self.cfg.target_swing_foot_forward_vel, 0.0, 1.0)
            * (1.0 - left_phase_lift_dense_reward)
            * upright_reward
            * phase_motion_gate
        )
        left_up_vel_discovery_reward = torch.clamp(left_foot_vel_w[:, 2] / 0.10, 0.0, 1.0)

        left_up_vel_discovery_term = (
        1.25
        * left_phase_swing_gate
        * phase_active_gate
        * right_contact
        * left_unload_discovery_reward
        * left_up_vel_discovery_reward
        * upright_reward
        * height_gate
        * heading_gate
        )

        left_drag_discovery_term = -0.75 * left_drag_discovery_penalty

        both_feet_contact = left_contact * right_contact
        #Reward movement following the walking gait and penalize standing still and not attempting to move forward or swing foot
        step_activity_reward = torch.clamp(0.55 * phase_lift_reward * phase_active_gate + 0.25 * phase_single_support_reward * phase_active_gate + 0.20 * forward_progress_reward, 0.0, 1.0,)
        static_stand_penalty = (1.0 - step_activity_reward) * both_feet_contact * upright_reward * phase_motion_gate
        static_stand_term = -self.cfg.penalty_static_stand * static_stand_penalty





        fallen = root_z < self.cfg.fall_height

        pose_term = self.cfg.rew_pose * pose_reward
        vel_term = self.cfg.rew_vel * vel_reward
        bc_term = self.cfg.rew_bc * bc_reward
        upright_term = self.cfg.rew_upright * upright_reward
        height_term = self.cfg.rew_height * height_reward
        standing_height_term = self.cfg.rew_standing_height * standing_height_reward
        alive_term = self.cfg.rew_alive * torch.ones_like(root_z)
        action_rate_term = -self.cfg.penalty_action_rate * action_rate_penalty
        joint_vel_term = -self.cfg.penalty_joint_vel * joint_vel_penalty
        fall_term = -self.cfg.penalty_fall * fallen.float()
        lateral_vel_term = -self.cfg.penalty_lateral_vel * lateral_vel_penalty
        base_ang_vel_term = -self.cfg.penalty_base_ang_vel * base_ang_vel_penalty
        side_tilt_term = -self.cfg.penalty_side_tilt * side_tilt_penalty

        low_height_term = -self.cfg.penalty_low_height * low_height_penalty
        knee_crouch_term = -self.cfg.penalty_knee_crouch * knee_crouch_penalty

        #Key rewards
        reward = (
            alive_term
            + pose_term
            + vel_term
            + bc_term
            + upright_term
            + height_term
            + standing_height_term
            + lower_body_gait_term
            + forward_vel_term
            + trailing_foot_recovery_term
            + swing_foot_clearance_term
            + swing_unload_term
            + dense_swing_air_term
            + proper_forward_swing_term
            + static_stand_term
            + any_foot_lift_term
            + toe_tap_term
            + spin_step_term
            + sustained_swing_air_term
            + lift_unload_term
            + phase_swing_lift_term
            + phase_forward_swing_term
            + phase_single_support_term
            + wrong_phase_lift_term
            + phase_swing_air_term
            + phase_missed_lift_term
            + phase_wrong_support_air_term
            + left_phase_lift_boost_term
            + left_phase_forward_boost_term
            + left_phase_right_support_term
            + left_support_lift_combo_term
            + left_step_touchdown_term
            + right_phase_lift_boost_term
            + right_phase_forward_boost_term
            + right_support_lift_combo_term
            + left_phase_right_air_term
            + left_phase_left_heavy_term
            + left_unload_when_right_ready_term
            + left_lift_when_right_ready_term
            + right_stance_for_left_term
            + right_re_lift_during_left_term
            + right_stance_short_for_left_term
            + left_up_vel_when_right_ready_term
            + left_knee_flex_when_right_ready_term
            + left_contact_when_right_ready_term
            + left_airborne_when_right_ready_term
            + left_air_fwd_when_right_ready_term
            + left_drag_during_left_swing_term
            + left_no_lift_when_right_ready_term
            + left_load_during_left_swing_term
            + left_lift_discovery_term
            + left_unload_discovery_term
            + left_drag_discovery_term
            + left_up_vel_discovery_term
            + right_no_lift_term
            + right_up_vel_when_left_ready_term
            + right_airborne_when_left_ready_term
            + left_stance_for_right_term
            + right_lift_discovery_term
            + right_up_vel_discovery_term
            + right_drag_discovery_term
            + right_step_touchdown_term
            + right_phase_missed_lift_soft_term
            + phase_airtime_hold_term
            + phase_air_forward_term
            + phase_high_lift_term
            + swing_lateral_term
            + forward_direction_term
            + foot_x_gap_term
            + weight_shift_term
            + single_support_term
            + foot_airtime_term
            + foot_slip_term
            + action_rate_term
            + joint_vel_term
            + lateral_vel_term
            + lateral_dominance_term
            + base_ang_vel_term
            + side_tilt_term
            + low_height_term
            + knee_crouch_term
            + knee_collapse_term
            + knee_asymmetry_term
            + backward_term
            + yaw_rate_term
            + fall_term
        )

        # Accumulate episode statistics for TensorBoard logging.
        self._episode_sums["pose"] += pose_term
        self._episode_sums["vel"] += vel_term
        self._episode_sums["bc"] += bc_term
        self._episode_sums["upright"] += upright_term
        self._episode_sums["height"] += height_term
        self._episode_sums["standing_height"] += standing_height_term
        self._episode_sums["lower_body_gait"] += lower_body_gait_term
        self._episode_sums["lower_body_error"] += lower_body_error
        self._episode_sums["alive"] += alive_term
        self._episode_sums["action_rate_penalty"] += action_rate_term
        self._episode_sums["joint_vel_penalty"] += joint_vel_term
        self._episode_sums["fall_penalty"] += fall_term
        self._episode_sums["total"] += reward
        self._episode_sums["episode_length"] += 1.0
        self._episode_sums["root_height"] += root_z
        self._episode_sums["pose_error"] += pose_error
        self._episode_sums["bc_residual_l2"] += bc_residual_l2
        self._episode_sums["lateral_vel_penalty"] += lateral_vel_term
        self._episode_sums["base_ang_vel_penalty"] += base_ang_vel_term
        self._episode_sums["side_tilt_penalty"] += side_tilt_term
        self._episode_sums["lateral_velocity"] += root_lin_vel_b[:, 1]
        self._episode_sums["base_ang_vel"] += torch.mean(torch.abs(root_ang_vel_b), dim=-1)
        self._episode_sums["low_height_penalty"] += low_height_term
        self._episode_sums["knee_crouch_penalty"] += knee_crouch_term
        self._episode_sums["mean_knee_angle"] += mean_knee_angle
        self._episode_sums["knee_collapse_penalty"] += knee_collapse_term
        self._episode_sums["max_knee_angle"] += max_knee_angle
        self._episode_sums["forward_vel"] += forward_vel
        self._episode_sums["forward_vel_reward"] += forward_vel_term
        self._episode_sums["backward_penalty"] += backward_term
        self._episode_sums["yaw_rate_penalty"] += yaw_rate_term
        self._episode_sums["left_knee_angle"] += knee_angles[:, 0]
        self._episode_sums["right_knee_angle"] += knee_angles[:, 1]
        self._episode_sums["knee_asymmetry"] += knee_asymmetry
        self._episode_sums["knee_asymmetry_penalty"] += knee_asymmetry_term
        self._episode_sums["trailing_foot_recovery"] += trailing_foot_recovery_term
        self._episode_sums["swing_foot_clearance"] += swing_foot_clearance_term
        self._episode_sums["trailing_foot_lift"] += trailing_foot_lift
        self._episode_sums["trailing_foot_fwd_vel"] += trailing_foot_fwd_vel
        self._episode_sums["foot_x_gap"] += foot_x_gap
        self._episode_sums["foot_x_gap_penalty"] += foot_x_gap_term
        self._episode_sums["left_contact"] += left_contact
        self._episode_sums["right_contact"] += right_contact
        self._episode_sums["left_force_z"] += left_force_z
        self._episode_sums["right_force_z"] += right_force_z
        self._episode_sums["weight_shift"] += weight_shift_term
        self._episode_sums["single_support"] += single_support_term
        self._episode_sums["foot_airtime"] += foot_airtime_term
        self._episode_sums["foot_slip_penalty"] += foot_slip_term
        self._episode_sums["left_air_time"] += self.left_air_time
        self._episode_sums["right_air_time"] += self.right_air_time
        self._episode_sums["raw_single_support"] += single_support_reward
        self._episode_sums["raw_foot_airtime"] += foot_airtime_reward
        self._episode_sums["raw_weight_shift"] += weight_shift_reward
        self._episode_sums["raw_foot_slip"] += foot_slip_penalty
        self._episode_sums["swing_unload"] += swing_unload_term
        self._episode_sums["static_stand_penalty"] += static_stand_term
        self._episode_sums["dense_swing_air"] += dense_swing_air_term
        self._episode_sums["any_foot_lift"] += any_foot_lift_term
        self._episode_sums["lift_unload"] += lift_unload_term
        self._episode_sums["proper_forward_swing"] += proper_forward_swing_term
        self._episode_sums["toe_tap_penalty"] += toe_tap_term
        self._episode_sums["spin_step_penalty"] += spin_step_term
        self._episode_sums["sustained_swing_air"] += sustained_swing_air_term
        self._episode_sums["counted_trailing_lift"] += counted_trailing_lift_reward
        self._episode_sums["phase_swing_lift"] += phase_swing_lift_term
        self._episode_sums["phase_forward_swing"] += phase_forward_swing_term
        self._episode_sums["phase_single_support"] += phase_single_support_term
        self._episode_sums["wrong_phase_lift"] += wrong_phase_lift_term
        self._episode_sums["phase_active_gate"] += phase_active_gate
        self._episode_sums["phase_swing_air"] += phase_swing_air_term
        self._episode_sums["phase_missed_lift"] += phase_missed_lift_term
        self._episode_sums["phase_wrong_support_air"] += phase_wrong_support_air_term
        self._episode_sums["left_phase_lift"] += left_phase_lift_reward * left_phase_active
        self._episode_sums["right_phase_lift"] += right_phase_lift_reward * right_phase_active
        self._episode_sums["left_phase_lift_dense"] += left_phase_lift_dense_reward * left_phase_active
        self._episode_sums["right_phase_lift_dense"] += right_phase_lift_dense_reward * right_phase_active
        self._episode_sums["left_phase_lift_boost"] += left_phase_lift_boost_term
        self._episode_sums["left_phase_forward_boost"] += left_phase_forward_boost_term
        self._episode_sums["left_phase_right_support"] += left_phase_right_support_term
        self._episode_sums["left_support_lift_combo"] += left_support_lift_combo_term
        self._episode_sums["left_step_touchdown"] += left_step_touchdown_term
        self._episode_sums["left_phase_right_air"] += left_phase_right_air_term
        self._episode_sums["left_phase_left_heavy"] += left_phase_left_heavy_term
        self._episode_sums["right_stance_for_left"] += right_stance_for_left_term
        self._episode_sums["right_re_lift_during_left"] += right_re_lift_during_left_term
        self._episode_sums["right_stance_time"] += right_new_stance_time * left_swing_intent_gate
        self._episode_sums["right_load_frac"] += right_load_frac 
        self._episode_sums["right_stance_short_for_left"] += right_stance_short_for_left_term
        self._episode_sums["left_unload_when_right_ready"] += left_unload_when_right_ready_term
        self._episode_sums["left_lift_when_right_ready"] += left_lift_when_right_ready_term
        self._episode_sums["left_up_vel_when_right_ready"] += left_up_vel_when_right_ready_term
        self._episode_sums["left_knee_flex_when_right_ready"] += left_knee_flex_when_right_ready_term
        self._episode_sums["left_contact_when_right_ready"] += left_contact_when_right_ready_term
        self._episode_sums["left_airborne_when_right_ready"] += left_airborne_when_right_ready_term
        self._episode_sums["left_air_fwd_when_right_ready"] += left_air_fwd_when_right_ready_term
        self._episode_sums["left_drag_during_left_swing"] += left_drag_during_left_swing_term
        self._episode_sums["left_no_lift_when_right_ready"] += left_no_lift_when_right_ready_term
        self._episode_sums["left_load_during_left_swing"] += left_load_during_left_swing_term
        self._episode_sums["left_lift_discovery"] += left_lift_discovery_term
        self._episode_sums["left_unload_discovery"] += left_unload_discovery_term
        self._episode_sums["left_drag_discovery"] += left_drag_discovery_term
        self._episode_sums["left_up_vel_discovery"] += left_up_vel_discovery_term
        self._episode_sums["raw_left_up_vel_discovery"] += left_up_vel_discovery_reward
        self._episode_sums["phase_airtime_hold"] += phase_airtime_hold_term
        self._episode_sums["phase_air_forward"] += phase_air_forward_term
        self._episode_sums["swing_lateral_penalty"] += swing_lateral_term
        self._episode_sums["forward_direction"] += forward_direction_term
        self._episode_sums["lateral_dominance_penalty"] += lateral_dominance_term
        self._episode_sums["phase_high_lift"] += phase_high_lift_term
        self._episode_sums["right_phase_lift_boost"] += right_phase_lift_boost_term
        self._episode_sums["right_phase_forward_boost"] += right_phase_forward_boost_term
        self._episode_sums["right_support_lift_combo"] += right_support_lift_combo_term
        self._episode_sums["right_no_lift_when_left_ready"] += right_no_lift_term
        self._episode_sums["right_up_vel_when_left_ready"] += right_up_vel_when_left_ready_term
        self._episode_sums["right_airborne_when_left_ready"] += right_airborne_when_left_ready_term
        self._episode_sums["right_step_touchdown"] += right_step_touchdown_term
        self._episode_sums["right_lift_discovery"] += right_lift_discovery_term
        self._episode_sums["right_up_vel_discovery"] += right_up_vel_discovery_term
        self._episode_sums["right_drag_discovery"] += right_drag_discovery_term
        self._episode_sums["raw_right_up_vel_discovery"] += right_up_vel_discovery_reward
        self._episode_sums["left_stance_for_right"] += left_stance_for_right_term
        self._episode_sums["left_stance_time_for_right"] += left_new_stance_time * right_phase_swing_gate * phase_active_gate
        self._episode_sums["left_load_frac_for_right"] += (left_load_frac_for_forward * right_phase_swing_gate * phase_active_gate)
        self._episode_sums["left_support_ready_for_right"] += left_support_ready_for_right
        self._episode_sums["right_lift_demand_gate"] += right_lift_demand_gate
        self._episode_sums["right_lift_discovery_gate"] += right_lift_discovery_gate
        self._episode_sums["right_phase_missed_lift_soft"] += right_phase_missed_lift_soft_term


        self.prev_actions = self.actions.clone()

        # To know how long each foot has been in contact with the floor
        self.left_stance_time = left_new_stance_time
        self.right_stance_time = right_new_stance_time

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



        # Start all envs from reference frame 0 first. Later, randomize this for robustness when learning.
        #Changed this to prevent overfitting first step, randomizing gait phase sequence slightly.
        phase0 = torch.rand(num_reset, device=self.device)
        ref_idx0 = torch.remainder((phase0 * self.num_ref_frames).long(), self.num_ref_frames)

        joint_pos = self.reference_q[ref_idx0]
        joint_vel = torch.zeros_like(joint_pos)

        default_root_state = self.robot.data.default_root_state[env_ids_t].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids_t]
        default_root_state[:, 2] = self.cfg.target_root_height

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids_t)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_t)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_t)

        self.phase[env_ids_t] = phase0
        self.actions[env_ids_t] = 0.0
        self.prev_actions[env_ids_t] = 0.0
        self.q_bc[env_ids_t] = joint_pos
        self.q_target[env_ids_t] = joint_pos

        #Reset air and contact time at the end
        self.left_air_time[env_ids_t] = 0.0
        self.right_air_time[env_ids_t] = 0.0
        self.prev_left_contact[env_ids_t] = 1.0
        self.prev_right_contact[env_ids_t] = 1.0
        self.left_stance_time[env_ids_t] = 0.0
        self.right_stance_time[env_ids_t] = 0.0
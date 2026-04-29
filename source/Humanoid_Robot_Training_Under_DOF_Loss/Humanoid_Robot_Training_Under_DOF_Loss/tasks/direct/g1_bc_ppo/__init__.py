import gymnasium as gym
'''Register the gymansium environment as Isaac-G1_BC-PPO-Walk-Direct-v0 for training and play rsl_rl script. Isaac Lab custom DirectRLEnv.
This is what will be trained and played in Isaac Lab.'''
gym.register(
    id="Isaac-G1-BC-PPO-Walk-Direct-v0",
    entry_point=(
        "Humanoid_Robot_Training_Under_DOF_Loss.tasks.direct.g1_bc_ppo.g1_bc_ppo_env:"
        "G1BCPPOEnv"
    ),

    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "Humanoid_Robot_Training_Under_DOF_Loss.tasks.direct.g1_bc_ppo.g1_bc_ppo_env:"
            "G1BCPPOEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "Humanoid_Robot_Training_Under_DOF_Loss.tasks.direct.g1_bc_ppo.agents.rsl_rl_ppo_cfg:"
            "G1BCPPORunnerCfg"
        ),
    },
)
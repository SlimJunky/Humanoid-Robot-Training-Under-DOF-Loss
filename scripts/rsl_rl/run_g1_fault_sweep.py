from __future__ import annotations

import subprocess
from pathlib import Path


TASK = "Isaac-G1-BC-PPO-Walk-Direct-v0"
CHECKPOINT = r"logs/rsl_rl/YOUR_EXPERIMENT/YOUR_RUN/model_400.pt"
OUT_CSV = "results/g1_fault_sweep.csv"
EPISODES = 10

JOINTS = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
]

TORQUE_SCALES = [1.0, 0.5, 0.25, 0.0]


def run(cmd: list[str]):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    Path("results").mkdir(exist_ok=True)

    # Baseline
    run([
        "python",
        "scripts/rsl_rl/eval_g1_metrics_faults.py",
        "--task", TASK,
        "--checkpoint", CHECKPOINT,
        "--num_envs", "1",
        "--episodes", str(EPISODES),
        "--fault_mode", "none",
        "--out_csv", OUT_CSV,
        "--headless",
    ])

    # Torque sweep
    for joint in JOINTS:
        for scale in TORQUE_SCALES:
            if scale == 1.0:
                continue

            run([
                "python",
                "scripts/rsl_rl/eval_g1_metrics_faults.py",
                "--task", TASK,
                "--checkpoint", CHECKPOINT,
                "--num_envs", "1",
                "--episodes", str(EPISODES),
                "--fault_mode", "torque",
                "--fault_joint", joint,
                "--torque_scale", str(scale),
                "--fault_time_s", "2.0",
                "--out_csv", OUT_CSV,
                "--headless",
            ])

    # Small lock sweep
    for joint in [
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
    ]:
        run([
            "python",
            "scripts/rsl_rl/eval_g1_metrics_faults.py",
            "--task", TASK,
            "--checkpoint", CHECKPOINT,
            "--num_envs", "1",
            "--episodes", str(EPISODES),
            "--fault_mode", "lock",
            "--fault_joint", joint,
            "--fault_time_s", "2.0",
            "--out_csv", OUT_CSV,
            "--headless",
        ])


if __name__ == "__main__":
    main()
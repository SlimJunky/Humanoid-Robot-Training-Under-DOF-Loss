# Fault-Tolerant Humanoid Motion: Evaluating Imitation Learning and Reinforcement Learning under Degree-of-Freedom Loss

This study investigates the robustness of humanoid locomotion policies trained using imitation learning and reinforcement learning in a simulated environment. The central research question is: **how does an on-policy reinforcement learning controller respond to partial actuator or degree-of-freedom failure, and which joints are most critical for maintaining stable locomotion?**

The project uses a minimal Unitree G1 humanoid asset in Isaac Lab. A weak Behavioral Cloning (BC) policy is first trained from selected retargeted walking demonstrations and then used as a weak prior for Proximal Policy Optimisation (PPO). Rather than learning locomotion entirely from scratch, the PPO controller learns residual joint corrections around the BC policy output, allowing it to improve stability, balance, and forward stepping behaviour initially. Then the PPO algorithm is pushed to its limits and fine tuned with rewards and penalty terms whilst moving away from matching the BC prior. The humanoid robot develops while maintaining a target forward velocity, upright posture and a survival of an entire trial episode without falling under Isaac Lab simulator physics.

Once a stable baseline locomotion policy is obtained, it is evaluated under controlled fault conditions. These include partial torque reduction and complete joint locking applied after a fixed point in the trial episode. The resulting behaviour is measured using metrics such as survival time, fall rate, root height, forward velocity, lateral drift, base angular velocity, joint-torque usage and more.

The objective  of this study is not to achieve perfect human walking or to match the selected motion-capture dataset perfectly but instead to evaluate the fault tolerance of a learned humanoid controller under degree-of-freedom loss. By comparing nominal and faulty runs, the experiments aim to identify which joints have the greatest effect on locomotion stability and determine the point at which actuator degradation leads to policy failure.

Below is some helpful suggestions on how to run important key scripts and how to run this external Isaac Lab project template onto your device. This repository is part of supporting evidence to validate the results from its accompanying dissertation paper.




## Isaac Lab NVIDIA external project template

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/Humanoid_Robot_Training_Under_DOF_Loss
    ```

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/Humanoid_Robot_Training_Under_DOF_Loss/Humanoid_Robot_Training_Under_DOF_Loss/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/Humanoid_Robot_Training_Under_DOF_Loss"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```

## Scripts

### Dump Unitree G1 Asset Information - g1_dump_asset_info.py

This script launches a minimal Isaac Lab scene, spawns the Unitree G1 robot using `G1_MINIMAL_CFG`, and exports useful robot asset information such as joint names, body names, default joint positions, velocity values, and joint limits.

The script is useful for checking the exact G1 joint order used by Isaac Lab before retargeting any motion capture data.

The output file will be saved to `outputs/g1_asset_dump.json` relative to the project root.

#### Run Command

```powershell
python scripts/g1_dump_asset_info.py --out outputs/g1_asset_dump.json
```


## Inspect AMASS Motion Files and Generate Manifest - inspect_amass_cmu.py

This script scans a folder of AMASS `.npz` motion files and creates a JSON and CSV manifest. It records useful information such as file name, relative path, motion label guess, FPS, number of frames, duration, and available AMASS keys.

The manifests are useful for selecting candidate clips and understanding the format of the data before preparing them for retargeting onto the Unitree G1 robot. Mainly used for debugging not part of the pipeline required to run the experiment.

### Run Command

```powershell
python scripts_amass/inspect_amass_cmu.py --input_dir data/selected_data/Walk --include_all
```

### Prepare AMASS Motion for Retargeting - amass_retarget_preparation.py

This script prepares a selected AMASS `.npz` motion file into a simpler retarget-ready format for later Unitree G1 mapping. It extracts the SMPL+H body pose, hand pose, root translation, root orientation, root quaternion, root yaw, betas, and timing information.

The script also resamples the motion to a target FPS, specifically 60FPS and optionally normalises the root X/Y translation so the motion starts from zero. The generated output is used as the intermediate motion file before creating the first-pass G1 mapped joint targets.

Below is an example of how this command was originally run with an expected location and default name within project repository as "data/selected_data/Walk/37_01_poses_slow_walk_retarget_ready.npz".

By default, the script normalizes the root X/Y translation so the motion starts at the origin, while keeping the original vertical root height. This makes the motion easier to use later for retargeting and playback in Isaac Lab.

The script generates a "retarget-ready" .npz file and a "metadata" .json file in the expected out-directory

Please make sure to obtain the right pose data 37_01_poses as SMPL+H from the AMASS version of the CMU MoCap Dataset.

#### Run Command

```powershell
python scripts_amass/amass_retarget_preparation.py data/selected_data/Walk/37_01_poses_slow_walk.npz --out-dir data/retarget_ready --target-fps 60
```

### Create First-Pass G1 Joint Targets from Retarget-Ready AMASS Motion - g1_map_walk_offline_pass.py

This script converts a retarget-ready AMASS motion file into a Unitree G1 joint mapping target file.

It uses the previously generated G1 asset dump to read the exact Isaac Lab G1 joint order, default joint positions, and soft joint limits. The script starts each frame from the G1 default joint pose, maps selected SMPL+H body joints onto matching G1 joints, clips the result to the G1 soft joint limits, and saves the result as a playback/training-ready `.npz` file.

This first-pass mapping is used before generating Behavioral Cloning demonstrations. Please use the playback script for a visual check and make sure the g1_asset_dump.json file and the retarget_ready.npz file generated from the "amass_retarget_prepartation".py script.

#### Run Command

```powershell
python scripts_amass/map_motion_to_g1_first_pass.py data/retarget_ready/Walk/37_01_poses_slow_walk_retarget_ready.npz --g1-dump-json outputs/g1_asset_dump.json --out-dir data/mapped
```

### Play Mapped G1 Motion in Isaac Lab

This script launches a simple Isaac Lab scene, spawns the Unitree G1 robot using `G1_MINIMAL_CFG`, and plays back a mapped G1 joint-target `.npz` file. It is used to visually inspect the first-pass retargeted motion before using it for BC demos generated in another script before BC policy prior

The script checks that the joint order in the mapped `.npz` file matches the current G1 articulation. It then sends each frame of `joint_targets` to the robot as position targets and steps the physics simulation.

This is a physics-based playback script so I recommend running this with the below command enabling frozen root to visually see mapped leg movement.

#### Run Command

```powershell
python scripts_amass/play_mapped_g1_motion.py data/mapped/Walk/37_01_poses_slow_walk_retarget_ready_g1_first_pass.npz --loop --reset-on-loop
```
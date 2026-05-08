# Fault-Tolerant Humanoid Motion: Evaluating Imitation Learning and Reinforcement Learning under Degree-of-Freedom Loss

This study investigates the robustness of humanoid locomotion policies trained using imitation learning and reinforcement learning in a simulated environment. The central research question is: **how does an on-policy reinforcement learning controller respond to partial actuator or degree-of-freedom failure, and which joints are most critical for maintaining stable locomotion?**

The project uses a minimal Unitree G1 humanoid asset in Isaac Lab. A weak Behavioral Cloning (BC) policy is first trained from selected retargeted walking demonstrations and then used as a weak prior for Proximal Policy Optimisation (PPO). Rather than learning locomotion entirely from scratch, the PPO controller learns residual joint corrections around the BC policy output, allowing it to improve stability, balance, and forward stepping behaviour initially following a simple imitation learning pipeline practice. Then the PPO algorithm is pushed to its limits and fine tuned with rewards and penalty terms whilst moving away from matching the BC prior. The humanoid robot develops while maintaining a target forward velocity, upright posture and a survival of an entire trial episode without falling under Isaac Lab simulator physics.

Once a stable baseline locomotion policy is obtained, it is evaluated under controlled fault conditions. These include partial torque reduction and complete joint locking applied after a fixed point in the trial episode. The resulting behaviour is measured using metrics such as survival time, fall rate, root height, forward velocity, lateral drift, base angular velocity, joint-torque usage and more.

The objective  of this study is not to achieve perfect human walking or to match the selected motion capture dataset perfectly but instead to evaluate the fault tolerance of a learned humanoid controller under degree-of-freedom loss. By comparing nominal and faulty runs, the experiments aim to identify which joints have the greatest effect on locomotion stability and determine the point at which actuator degradation leads to policy failure.

Below is some helpful suggestions on how to run all the scripts in this repository and how to generate the data needed to repeat the experiments shown in the study. There is also a description of the environment and its supporting packages to be able to run this external isaac lab project aswell as the default provided install instructions. 

This repository is part of supporting evidence to validate the results from its accompanying dissertation paper.

## Hardware & Software Environment Details

The experiments were developed and tested using Isaac Sim / Isaac Lab with GPU acceleration. The project was run from a Windows external Isaac Lab project, with Robomimic Behavioral Cloning training performed separately in a Linux Ubuntu/WSL Ubuntu Python environment.


### Hardware Used:

- Operating system: Microsoft Windows 10 Pro, Build 19045
- System manufacturer/model: MSI MS-7915
- CPU: Intel Core i7-4790K @ 4.00 GHz
- CPU cores/threads: 4 cores / 8 logical processors
- RAM: 24 GB
- GPU: NVIDIA GeForce RTX 2080 SUPER
- GPU VRAM: 8 GB
- NVIDIA driver version: 591.86
- CUDA driver version reported by NVIDIA-SMI: 13.1


### Used Software Packages & Conda Environments:

```text
- Recommended miniconda environment named as default "env_isaaclab" with all accompanying packages
- NVIDIA Drivers >= 580.88 version
- NVIDIA Isaac Sim 5.1.0-rc.19, any Isaac Sim 5.X should work
- NVIDIA Isaaclab latest cloned repository from main branch, Isaac Lab 0.5.0.
- isaaclab_rl-0.50 (comes with download)
- isaaclab_tasks-0.11.14 (comes with download)
- Python 3.11.15, any Python 3.11.X should work
- PyTorch 2.7.0 with GPU acceleration enabled via CUDA 12.8 or better
- torch-2.7.0+cu128
- torchvision-0.22.0+cu128
- torchaudio-2.7.0+cu128
- RSL-RL library for PPO training during external project download
- TensorBoard package for training log visualisation

If you wish to recreate the BC imitation learning pipeline yourself this is possible by having the same packages and environment as shown below and also following the installation & documentation guide provided by the official Robo-mimic landing page. This is not required to view the stable policy controller or run the experiments described in the study.

- Recommended miniconda environment within a Linux OS or through WSL named as default "robomimic_venv" with all accompanying packages
- Robo-mimic v0.5 (latest) for Behavioral Cloning
- Python 3.8.0
- Pytorch 2.0.0, Wheels such as torch-2.0.0+cu118, torchvision-01.15.1+cu118, torchaudio-2.0.1+cu118
- Recommended CMake 3.31 < CMake 4.X. Sometimes there is an issue with egl_probe within the robo_mimic install if CMake is not the older version.

```
Install robomimic following this guide for Linux installing robomimic from source and using the recommended PyTorch:

```text
https://robomimic.github.io/docs/introduction/installation.html
```
No optional installations or datasets are required.


## Recommended Install Route

To reproduce this project, first install Isaac Sim and Isaac Lab using the official NVIDIA documentation rather than installing packages manually one-by-one. The guides are available here:

```text
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
```

Isaac Lab API which helped in designing and producing the code in this repository
```text
https://isaac-sim.github.io/IsaacLab/main/source/api/index.html
```


In this README is a provided default generated template for debugging the install below, please use the official NVIDIA repositories and documentation to debug the installation beyond this.

Recommended setup order:

1. Check the Isaac Sim 5.1 hardware and driver requirements. Check the packages & environment dependencies above
2. Install Isaac Sim 5.1 using the recommended Python/pip installation route.
3. Install Isaac Lab using the official Isaac Lab local installation guide.
4. Clone this repository separately from the main Isaac Lab repository as an external Isaac Lab project.
5. Activate the Isaac Lab Python/conda environment and install this project in editable mode from the repository root as such:

```bash
python -m pip install -e source/Humanoid_Robot_Training_Under_DOF_Loss
```

This repository was originally created from the Isaac Lab external project template generator. The important template choices for this project were:

```text
Project type: External Isaac Lab project
Workflow: Direct RL environment
RL library: RSL-RL
Training algorithm: PPO
Task style: Single-agent reinforcement learning
Main environment: G1BCPPOEnv
Main task name: Isaac-G1-BC-PPO-Walk-Direct-v0
```

To verify tasks are available to run after the project is installed use:

```powershell
python scripts/list_envs.py
```


## Obtaining AMASS SMPL+H Motion Capture Data

Please go to the official AMASS website:

https://amass.is.tue.mpg.de/index.html

Then please register your account and accept the license / terms of service when obtaining the data following the BSD 3-Clause. 

Go into the Downloads section and look for the CMU dataset. Then download the "SMPL+H G" and accept the license. Then unpack the CMU.tar.bz2 zip.

Then look into this directory and obtain this .npz file:

```text
37\37_01_poses.npz
```

## Minimum Data Processing Pipeline Required

After obtaining the chosen AMASS / SMPL+H `.npz` walking motion file, the minimum processing pipeline needed for this project is:

```text
AMASS SMPL+H .npz
retarget-ready AMASS .npz
mapped Unitree G1 joint-target .npz
Robomimic-style BC dataset metadata
```
Use the script information provided in this README to achieve this data processing or alternatively run these commands in tandem and place the resulting files in the correct location. This is required to be done locally and could not be pre-processed and re-distributed in order to keep within the BSD3-Clause that prevents the distribution of the SMPL+H motion capture data, modified or not, in a third-party area.


### 1

```powershell
python scripts/g1_dump_asset_info.py --out outputs/g1_asset_dump.json
```

Before mapping the AMASS motion file onto Unitree G1, generate a G1 asset dump. This records the exact Isaac Lab joint order, default joint positions, body names, and joint limits used by `G1_MINIMAL_CFG`.

### 2
```powershell
python scripts_amass/amass_retarget_preparation.py data/selected_data/Walk/37_01_poses_slow_walk.npz --out-dir data/retarget_ready --target-fps 60
```
Converts your obtained 37_01.npz file renamed as 37_01_poses_slow_walk inside of data/selected_data/Walk into cleaner retarget-ready format at 60FPS

### 3
```powershell
python scripts/g1_map_walk_offline_pass.py data/retarget_ready/Walk/37_01_poses_slow_walk_retarget_ready.npz --g1-dump-json outputs/g1_asset_dump.json --out-dir data/mapped
```
Converts retarget-ready AMASS motion into minimal Unitree G1 joint targets using the G1 asset dump and soft joint limits.

### 4
```powershell
python scripts/g1_record_bc_reference_demos.py data/mapped/Walk/37_01_poses_slow_walk_retarget_ready_g1_first_pass.npz --out-hdf5 data/bc_dataset_demonstrations/g1_walk_reference_bc_1024_regular.hdf5 --num-demos 1024 --include-phase --obs-noise-std 0.005 --action-noise-std 0.0 --speed-jitter 0.05 --seed 0
```
This creates the robomimic-style HDF5 dataset and JSON metadata used by the BC prior. Specifically the metadata from the JSON is used for the "G1BCPPOEnv direct environment" for playback. It tells the environment script information such as the final policy G1 joint order, action lower and upper bounds and how to denormalize BC prior output.

It also includes the location of the mapped .npz file generated in step 3 for reference.


## Evaluating Policy Controller & Running Experiments

### Run the Final Stable Policy Controller used in the experiments:

```powershell
python scripts/rsl_rl/play.py --task Isaac-G1-BC-PPO-Walk-Direct-v0 --num_envs 1 --checkpoint PPO_RL_policy_checkpoints/PPO_WALK_GOOD_FINAL/model_11992.pt --device cuda:0
```

TO-DO: Put the key scripts here not much context. Scripts for experiment especially


## Isaac Lab NVIDIA external project template - Default Recommendation as provided by NVIDIA

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
python scripts/g1_map_walk_offline_pass.py data/retarget_ready/Walk/37_01_poses_slow_walk_retarget_ready.npz --g1-dump-json outputs/g1_asset_dump.json --out-dir data/mapped
```

### Preview Mapped G1 Motion in Isaac Lab - g1_preview_mapped_motion.py

This script launches a simple Isaac Lab scene, spawns the Unitree G1 robot using `G1_MINIMAL_CFG`, and plays back a mapped G1 joint-target `.npz` file. It is used to visually inspect the first-pass retargeted motion before using it for BC demos generated in another script before BC policy prior

The script checks that the joint order in the mapped `.npz` file matches the current G1 articulation. It then sends each frame of `joint_targets` to the robot as position targets and steps the physics simulation.

This is a physics-based playback script so I recommend running this with the below command enabling fixed root to visually see mapped leg movement.

#### Run Command

```powershell
python scripts/g1_preview_mapped_motion.py data/mapped/Walk/37_01_poses_slow_walk_retarget_ready_g1_first_pass.npz --root-height 0.95 --loop --fps 60
```

### Generate BC Reference Demonstrations - generate_bc_reference_demos.py

This script creates a Robomimic-style HDF5 dataset from a mapped G1 joint-target `.npz` file. It randomizes the starting phase of the mapped walking motion to generate many BC demonstration episodes for training the behavioral cloning prior.

The observations contain G1 joint positions, joint velocities, and optionally normalized phase. For this project, `--include-phase` should be used so the observation is `q(37) + qd(37) + phase(1) = 75`.

The actions are next-frame G1 joint targets, normalized to `[-1, 1]` by default using the soft joint limits stored in the mapped `.npz`.

Please include these parameters to replicate the HDF5 BC demonstration episodes that were used in training the bc prior:

#### Run Command

```powershell
python scripts/g1_record_bc_reference_demos.py data/mapped/Walk/37_01_poses_slow_walk_retarget_ready_g1_first_pass.npz --out-hdf5 data/bc_dataset_demonstrations/g1_walk_reference_bc_1024_regular.hdf5 --num-demos 1024 --include-phase --obs-noise-std 0.005 --action-noise-std 0.0 --speed-jitter 0.05 --seed 0
```

### Add Robomimic Train/Validation Masks - add_robomimic_train_valid_mask.py

This script edits an existing Robomimic style `.hdf5` dataset and adds the expected train and validation masks under the `/mask` group. These masks tell Robomimic which demonstration episodes should be used for training and which should be used for validation.

For this project, the dataset was split using a `0.9` train ratio and `0.1` validation ratio. The script updates the HDF5 file in place and does not create a new dataset.

This script was made to fix an oversight I missed before doing BC imitation learning using Robo-mimic.

##### Run Command

```bash
python scripts_robomimic/add_robomimic_train_valid_mask.py data/bc_dataset_demonstrations/g1_walk_reference_bc_1024_regular.hdf5 --train-ratio 0.9 --valid-ratio 0.1 --seed 1 --overwrite
```

### Train Behavioral Cloning Policy with Robomimic - train.py [robomimic cloned repo]

This step trains the weak Behavioral Cloning policy used later as the prior for PPO. The Robomimic config file I used is an edited version based off the default provided by the robo-mimic repository. These are available here:

The config file structure and documentation used for this project is provided in: https://robomimic.github.io/docs/modules/configs.html

The default bc configuration file is normally inside the official robo-mimic repository 

```text
robomimic/exps/templates/bc.json
```
This can also be generated as a fresh default BC config JSON as such in bash:

```bash
python -c "from robomimic.config import config_factory; c = config_factory(algo_name='bc'); c.dump(filename='bc_default.json')"
```

I provide the correctly edited configuration file used for imitation learning training within this repository, but it is expected you move this file to the robomimic environment within Linux / WSL in your own file and change the path to match your project root:

```text
bc_config_files/g1_bc_walk.json
```

This script points to the generated G1 walking HDF5 dataset, uses the `train` and `valid` masks, and trains a low-dimensional BC policy from proprioceptive observations.

Please only change the config file "g1_bc_walk.json" to point at the name of your generated BC demonstrations HDF5 file within the "train" section of the JSON and your appropriate output directory.

For this project, the observation input was `q(37) + qd(37) + phase(1) = 75`, and the action output was the next-frame G1 joint target with `37` action dimensions. Actions were already normalized to `[-1, 1]` during BC dataset generation.

The model was trained in the separate Robomimic/WSL environment, not directly inside Isaac Lab. The resulting checkpoint was later exported to TorchScript and used as the BC prior inside the Isaac Lab PPO environment.

This is an unnecessary step since I provide the TorchScript Policy to recreate the experiment. However this is required if you wish to recreate the BC prior training shown briefly in the study.

#### Run Command

```bash
python robomimic/scripts/train.py --config <PATH_TO_REPO>/bc_config_files/g1_bc_walk.json --debug
```

### Export Robomimic BC Policy to TorchScript - export_bc_policy_torchscript.py

This script exports a trained Robomimic Behavioral Cloning `.pth` checkpoint into a standalone TorchScript `.pt` policy. The exported policy takes a 75-dimensional proprioceptive observation, `q(37) + qd(37) + phase(1)`, and outputs a normalized 37-dimensional G1 joint action.

This is used to move the trained BC prior from the Robomimic/WSL environment into the Isaac Lab PPO environment without requiring Robomimic to be installed inside Isaac Lab. This is because Robo-mimic for windows tends to fail without weird environment wrappers.

The output `.pt` path is provided manually as the second command argument. The final BC prior is used as described in the study. This is "epoch 81" which had the lowest validation loss and best training results viewed in TensorBoard.

#### Run Command

```bash
python scripts_robomimic/export_bc_policy_torchscript.py bc_walking_policy_checkpoints/g1_bc_walk/models/model_epoch_81_best.pth bc_walking_policy_checkpoints/g1_bc_walk/g1_model_epoch_81_best.pt --device cuda --obs-dim 75
```

### Play Exported BC Policy in Isaac Lab - g1_play_bc_policy.py

This script plays the exported TorchScript Behavioral Cloning policy inside Isaac Lab using the Unitree G1 robot. It rebuilds the same proprioceptive observation used during BC training, `q(37) + qd(37) + phase(1)`, runs the BC policy, denormalizes the output action back into G1 joint targets, and sends those targets to the robot.

It is mainly used to visually inspect the weak BC prior before PPO training. The script can also freeze the root for easier inspection, or play the mapped reference actions directly for comparison reusing similar logic as previous scripts

To run this it is recommended to have the "g1_walk_reference_bc_1024_regular.json" or similar file generated from "g1_record_bc_reference_demos.py" within the right folder directory

```text
data/bc_dataset_demonstrations/g1_walk_refernce_bc_1024_regular
```

The script uses the BC metadata JSON to load the correct joint order, action bounds, observation size, action size, and mapped reference `.npz` path. The Robomimic training config is not used here. This script needs the generated BC runtime metadata JSON.

#### Run Command

```powershell
python scripts/g1_play_bc_policy.py bc_walking_policy_checkpoints/g1_bc_walk/g1_model_epoch_81_best.pt data/bc_dataset_demonstrations/g1_walk_reference_bc_1024_regular.json --root-height 0.78 --gait-period-s 4.25 --control-decimation 2 --debug-every 120
```

### PPO Walking Environment - g1_bc_ppo_env.py

The main reinforcement learning environment is defined in:

```text
source/Humanoid_Robot_Training_Under_DOF_Loss/Humanoid_Robot_Training_Under_DOF_Loss/tasks/direct/g1_bc_ppo/g1_bc_ppo_env.py
```

### Train PPO Walking Policy - train.py

This command trains the PPO residual walking policy using RSL-RL. The task loads `G1BCPPOEnv`, the TorchScript BC prior, and the mapped walking reference motion from the project relative paths defined in `G1BCPPOEnvCfg`.

You can manage the amount of iterations and the number of parallel environments if your system supports this with the default commands provided by Isaac Lab.

The results of the training run by default will go into:

```text
logs\rsl_rl\g1_bc_ppo_walk
```

The recommendation is to run this script always with --headless in order to improve performance.

#### Run Command

```powershell
python scripts/rsl_rl/train.py --task Isaac-G1-BC-PPO-Walk-Direct-v0 --num_envs 512 --headless --max_iterations 12000
```

This is an example of how you would continue training from a policy checkpoint, by directly stating the location of the .pt file with resume_path argument:

```powershell
python scripts\rsl_rl\train.py  --task Isaac-G1-BC-PPO-Walk-Direct-v0  --num_envs 512 --max_iterations 600 --headless  --resume --resume_path <PATH_TO_YOUR_MODEL>
```

### Watch Training Progress - TensorBoard

If you have followed the install correctly with dependent packages then you should be able to use TensorBoard within this external Isaac Lab project.

You can run look at an "O" file by binding the generated "O" file results from a specified folder as such:

```powershell
tensorboard --logdir logs/rsl_rl
```

To see all of the key training checkpoint milestones please look inside here and follow the same command with a path to the chosen training evidence folder:

```text
training_evidence\tensorboard_runs
```

### Play Trained PPO Policy - play.py

This command plays back a trained PPO checkpoint in Isaac Lab. Use this after training to visually inspect the final locomotion policy without fault injection.

Replace `<RUN_FOLDER>` and `<CHECKPOINT>` with the actual log folder and model checkpoint created during training. 

However, the actual configuration values including rewards and penalty terms must match during playback the same way as it was trained. If you are planning to run the final policy checkpoint used for the experiments, "model_11992.pt" please keep the "g1_bc_ppo_env.py" file as provided with its configuration terms.

To check the final stable BC-Policy controller used for the experiments within the study please run this from here:

```text
PPO_RL_policy_checkpoints\PPO_WALK_GOOD_FINAL\model_11992.pt
```

#### Run Command

```powershell
python scripts/rsl_rl/play.py --task Isaac-G1-BC-PPO-Walk-Direct-v0 --num_envs 1 --checkpoint logs/rsl_rl/<RUN_FOLDER>/<CHECKPOINT>.pt
```

OR DIRECT FINAL POLICY CONTROLLER COMMAND:

```powershell
python scripts/rsl_rl/play.py --task Isaac-G1-BC-PPO-Walk-Direct-v0 --num_envs 1 --checkpoint PPO_RL_policy_checkpoints/PPO_WALK_GOOD_FINAL/model_11992.pt --device cuda:0
```


## Third-Party Data and Software Citations

This repository uses several third-party datasets, tools, and simulation frameworks. Please cite or acknowledge the following sources when using or extending this project:

```text
* AMASS: Archive of Motion Capture as Surface Shapes
* CMU Graphics Lab Motion Capture Database
* NVIDIA Isaac Sim
* NVIDIA Isaac Lab
* Robomimic
* PyTorch
* RSL-RL, where relevant for PPO training
```

This repository cites all references within this file:

```text
docs\references.bib
```

### Motion Capture Data

The motion-capture data used in this project was obtained from the AMASS dataset, using SMPL+H motion files derived from the CMU Graphics Lab Motion Capture Database.

Please cite AMASS:

```bibtex
@inproceedings{AMASS:2019,
  title={AMASS: Archive of Motion Capture as Surface Shapes},
  author={Mahmood, Naureen and Ghorbani, Nima and F. Troje, Nikolaus and Pons-Moll, Gerard and Black, Michael J.},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  year={2019},
  month={Oct},
  url={https://amass.is.tue.mpg.de},
  month_numeric={10}
}
```
### Acknowledgements

CMU MoCap Acknowledgment:

The data used in this project was obtained from mocap.cs.cmu.edu.
The database was created with funding from NSF EIA-0196217.


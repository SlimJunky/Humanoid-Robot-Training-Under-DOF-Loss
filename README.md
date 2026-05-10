# Fault-Tolerant Humanoid Motion: Evaluating Imitation Learning and Reinforcement Learning under Degree-of-Freedom Loss

This study investigates the robustness of humanoid locomotion policies trained using imitation learning and reinforcement learning in a simulated environment. The central research question is: **how does an on-policy reinforcement learning controller respond to partial actuator or degree-of-freedom failure, and which joints are most critical for maintaining stable locomotion?**

The project uses a minimal Unitree G1 humanoid asset in Isaac Lab. A weak Behavioral Cloning (BC) policy is first trained from selected retargeted walking demonstrations and then used as a weak prior for Proximal Policy Optimisation (PPO). Rather than learning locomotion entirely from scratch, the PPO controller learns residual joint corrections around the BC policy output, allowing it to improve stability, balance, and forward stepping behaviour initially following a simple imitation learning pipeline practice. Then the PPO algorithm is pushed to its limits and fine tuned with rewards and penalty terms whilst moving away from matching the BC prior. The humanoid robot develops while maintaining a target forward velocity, upright posture and a survival of an entire trial episode without falling under Isaac Lab simulator physics.

Once a stable baseline locomotion policy is obtained, it is evaluated under controlled fault conditions. These include partial torque reduction and complete joint locking applied after a fixed point in the trial episode. The resulting behaviour is measured using metrics such as survival time, fall rate, root height, forward velocity, lateral drift, base angular velocity, joint-torque usage and more.

The objective  of this study is not to achieve perfect human walking or to match the selected motion capture dataset perfectly but instead to evaluate the fault tolerance of a learned humanoid controller under degree-of-freedom loss. By comparing nominal and faulty runs, the experiments aim to identify which joints have the greatest effect on locomotion stability and determine the point at which actuator degradation leads to policy failure.

Below is some helpful suggestions on how to run all the scripts in this repository and how to generate the data needed to repeat the experiments shown in the study. There is also a description of the environment and its supporting packages to be able to run this external isaac lab project aswell as the default provided install instructions. 

This repository is part of supporting evidence to validate the results from its accompanying dissertation paper. Please look at the bottom of this readme to find all citations. 

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
```

If you wish to recreate the BC imitation learning pipeline yourself this is possible by having the same packages and environment as shown below and also following the installation & documentation guide provided by the official Robo-mimic landing page. This is not required to view the stable policy controller or run the experiments described in the study.

```text
- Recommended miniconda environment within a Linux OS or through WSL named as default "robomimic_venv" with all accompanying packages
- Robo-mimic v0.5 (latest) for Behavioral Cloning
- Python 3.8.0
- Pytorch 2.0.0, Wheels such as torch-2.0.0+cu118, torchvision-01.15.1+cu118, torchaudio-2.0.1+cu118
- Recommended CMake 3.31 < CMake 4.X. Sometimes there is an issue with egl_probe within the robo_mimic install if CMake is not the older version.

```
Install robomimic following this guide for Linux installing robomimic from source and using the recommended PyTorch:

https://robomimic.github.io/docs/introduction/installation.html

No optional installations or datasets are required.


## Recommended Install Route

To reproduce this project, first install Isaac Sim and Isaac Lab using the official NVIDIA documentation rather than installing packages manually one-by-one. The guides are available here:


https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html


Isaac Lab API which helped in designing and producing the code in this repository:

https://isaac-sim.github.io/IsaacLab/main/source/api/index.html



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
Use the script information provided in "SCRIPTS.md" to achieve this data processing or alternatively run these commands below in tandem for the same order as I have run for this study and place all the resulting files in the correct locations. This is required to be done locally and could not be pre-processed and re-distributed in order to keep within the BSD3-Clause that prevents the distribution of the SMPL+H motion capture data where possible.

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


## Evaluating Policy Controller & Running Experiments:

### Run The Final Stable Policy Controller By Itself:

```powershell
python scripts/rsl_rl/play.py --task Isaac-G1-BC-PPO-Walk-Direct-v0 --num_envs 1 --checkpoint PPO_RL_policy_checkpoints/PPO_WALK_GOOD_FINAL/model_11992.pt --device cuda:0
```

### Run The Experiments Against Final Stable Policy Controller:

Please find the provided script in:

```text
scripts\rsl_rl\eval_g1_metrics_experiment.py
```

Run this script from the project root directory as usual.

This script evaluates the final trained PPO walking checkpoint acting as the stable motion locomotion controller in Isaac Lab and writes the experiment results to CSV files. Use this script to validate the final nominal, joint-lock, and torque-reduction experiments used for the fault-tolerance analysis within this study.

A reminder the final stable PPO policy checkpoint used in the study is:

```text
PPO_RL_policy_checkpoints\PPO_WALK_GOOD_FINAL\model_11992.pt
```
The script runs the policy in an 8 second simulation episode for a selected number of episodes. Applies the chosen fault condition after 2.0 seconds and records lots of key metrics to a csv such as fall rate, timeout rate, episode duration, forward distance, root height, lateral drift, torque values and before/after actuator fault comparisons.

The script can run in three main modes:

```text
none    = no fault, used as the nominal baseline
lock    = locks a selected joint at its current angle after the chosen fault time. Lock epsilon is always 0.001 in the analysed experiment results
torque  = reduces the selected joint torque limit after the chosen fault time. The experiment uses 0.50 to 0.00 as analysis
```

This runs the final PPO policy with no injected fault. The --fault_time_s 2.0 value is still used as a reference split point for before/after metrics:

```powershell
python scripts/rsl_rl/eval_g1_metrics_experiment.py --task Isaac-G1-BC-PPO-Walk-Direct-v0 --checkpoint "PPO_RL_policy_checkpoints/PPO_WALK_GOOD_FINAL/model_11992.pt" --episodes 10 --fault_mode none --fault_time_s 2.0 --out_csv results_experiment/g1_policy_eval.csv --headless --disable_fabric --debug
```

This runs the final PPO policy and locks the chosen joint after 2.0 seconds. This example locks the left knee joint. Lock epsilon is always "0.001" for these experiments

```powershell
python scripts/rsl_rl/eval_g1_metrics_experiment.py --task Isaac-G1-BC-PPO-Walk-Direct-v0 --checkpoint "PPO_RL_policy_checkpoints/PPO_WALK_GOOD_FINAL/model_11992.pt" --episodes 10 --fault_mode lock --fault_joint left_knee_joint --fault_time_s 2.0 --lock_epsilon 0.001 --out_csv results_experiment/g1_policy_eval.csv --headless --disable_fabric --debug
```

This runs the final PPO policy and reduces the torque limit of the chosen joint after 2.0 seconds. A --torque_scale of 0.0 represents complete torque loss, while 0.5 would represent 50% available torque.

```powershell
python scripts/rsl_rl/eval_g1_metrics_experiment.py --task Isaac-G1-BC-PPO-Walk-Direct-v0 --checkpoint "PPO_RL_policy_checkpoints/PPO_WALK_GOOD_FINAL/model_11992.pt" --episodes 10 --fault_mode torque --fault_joint left_knee_joint --torque_scale 0.25 --fault_time_s 2.0 --out_csv results_experiment/g1_policy_eval.csv --headless --disable_fabric --debug
```
All experiment results that were run as part of the study are in "g1_policy_eval.csv" for raw episodic data & "g1_policy_eval_summary.csv". Each run of the command is appending the results onto these csv files.

For visual debugging run any of these commands with --episodes 1, remove --headless and make sure to add --debug. Visual runs are placed into "results_experiment\g1_policy_eval_visual.csv" for raw episodic data & "results_experiment\g1_policy_eval_visual_summary.csv" for summarized data during these runs.

### Available fault joints Unitree G1 Minimal Asset:

The `--fault_joint` argument must match one of the joint names used by the Unitree G1 minimal robot configuration. A reference list of all available joint names is provided in:

```text
docs/g1_minimal_fault_joint_names.txt
```

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

A detailed script-by-script guide is provided here:

```text
docs/SCRIPTS.md
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

Full BibTex citation entries are provided in:

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



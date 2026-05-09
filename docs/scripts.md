
## Project Scripts Information

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

OR DIRECT FINAL POLICY CONTROLLER COMMAND PROVIDED:

```powershell
python scripts/rsl_rl/play.py --task Isaac-G1-BC-PPO-Walk-Direct-v0 --num_envs 1 --checkpoint PPO_RL_policy_checkpoints/PPO_WALK_GOOD_FINAL/model_11992.pt --device cuda:0
```

### Evaluate Trained PPO Policy Under Fault Conditions - eval_g1_metrics_experiment.py

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

This runs the final PPO policy and locks the chosen joint after 2.0 seconds. This example locks the left knee joint. 

```powershell
python scripts/rsl_rl/eval_g1_metrics_experiment.py --task Isaac-G1-BC-PPO-Walk-Direct-v0 --checkpoint "PPO_RL_policy_checkpoints/PPO_WALK_GOOD_FINAL/model_11992.pt" --episodes 10 --fault_mode lock --fault_joint left_knee_joint --fault_time_s 2.0 --lock_epsilon 0.001 --out_csv results_experiment/g1_policy_eval.csv --headless --disable_fabric --debug
```

This runs the final PPO policy and reduces the torque limit of the chosen joint after 2.0 seconds. A --torque_scale of 0.0 represents complete torque loss, while 0.5 would represent 50% available torque.

```powershell
python scripts/rsl_rl/eval_g1_metrics_experiment.py --task Isaac-G1-BC-PPO-Walk-Direct-v0 --checkpoint "PPO_RL_policy_checkpoints/PPO_WALK_GOOD_FINAL/model_11992.pt" --episodes 10 --fault_mode torque --fault_joint left_knee_joint --torque_scale 0.0 --fault_time_s 2.0 --out_csv results_experiment/g1_policy_eval.csv --headless --disable_fabric --debug
```
All experiment results that were run as part of the study are in "g1_policy_eval.csv" for raw episodic data & "g1_policy_eval_summary.csv". Each run of the command is appending the results onto these csv files.

For visual debugging run any of these commands with --episodes 1, remove --headless and make sure to add --debug. Visual runs are placed into "results_experiment\g1_policy_eval_visual.csv" for raw episodic data & "results_experiment\g1_policy_eval_visual_summary.csv" for summarized data during these runs.

### Available fault joints Unitree G1 Minimal Asset:

The `--fault_joint` argument must match one of the joint names used by the Unitree G1 minimal robot configuration. A reference list of all available joint names is provided in:

```text
docs/g1_minimal_fault_joint_names.txt
```


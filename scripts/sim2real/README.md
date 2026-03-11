# Flexiv Rizon 4s - Sim2Real Gear Assembly Task

This repository contains the deployment pipeline for a Reinforcement Learning (RL) based gear assembly task using the Flexiv Rizon 4s robotic arm. The codebase bridges policies trained in simulation (Isaac Sim) to the real world using ROS 2, Pinocchio for differential inverse kinematics (IK), and a custom PyTorch JIT policy.

## 📁 Repository Structure

* `run_assembly_task.py`: Main control node. Handles state aggregation, force/torque sensor tare, policy inference, and differential IK.
* `robots/`: Contains robot URDFs, kinematics definitions, and the `assembly.py` wrapper which loads the PyTorch JIT policy and computes target twists.
* `policies/`: Stores the trained `.pt` models and their corresponding configurations.
* `utils/`: 
  * `sim2sim_pub.py`: Isaac Sim script to publish native physics wrench data.
  * `sim_adapter.py`: ROS 2 node that bridges standard simulation topics to `flexiv_msgs`.
* `requirements/`: 
  * `env_isaaclab.yml`: Packages list for the environment to run the simulator.
  * `ros_rl_env.yml`: Packages list for the environment to run the `run_assembly.py` and `sim_adapter.py` scripts.

---

## 🛠️ Prerequisites

* **OS:** Ubuntu 22.04 (Recommended)
* **ROS 2:** Jazzy
* **Simulation:** NVIDIA Isaac Sim ^5.0.0
* **Environments:** See `requirements/` folder above.
* **Workspaces:** A compiled `flexiv_ros2_ws` containing `flexiv_msgs` and `flexiv_bringup`.

---

## 🌱 Experimental Seeds

To evaluate the policy robustness, we use four different starting seeds that alter the relative starting position of the TCP (Tool Center Point) compared to the fixed goal position. The robot's initial joint configuration remains constant across all seeds:  
`[-27.46, -45.05, 52.05, 92.91, -39.24, 28.07, -150.73]` (degrees).

### Seed 0 (Base Position)
* **Fixed Position:** `[0.65091, 0.04118, 0.11824]`
* **TCP Position:** `[0.65978, 0.0401, 0.1599]`
* **Relative Start:** `[8mm, -1mm, 41mm]`
* **Distance to Goal:** `4.26 cm`

### Seed 1 (X Displacement)
* **Fixed Position:** `[0.64091, 0.04118, 0.11824]`
* **TCP Position:** `[0.65978, 0.0401, 0.1599]`
* **Relative Start:** `[19mm, -1mm, 41mm]` (~2cm offset in X)
* **Distance to Goal:** `4.58 cm`

### Seed 2 (Y Displacement)
* **Fixed Position:** `[0.65091, 0.02016, 0.11824]`
* **TCP Position:** `[0.65978, 0.0401, 0.1599]`
* **Relative Start:** `[8mm, 19mm, 41mm]` (~2cm offset in Y)
* **Distance to Goal:** `4.71 cm`

### Seed 3 (Z Displacement)
* **Fixed Position:** `[0.65091, 0.04118, 0.12824]`
* **TCP Position:** `[0.65978, 0.0401, 0.1599]`
* **Relative Start:** `[8mm, 1mm, 31mm]` (~3cm offset in Z)

---

## 🧠 Training & Exporting the Policy

Training is performed using `rl_games` via the `isaaclab.sh` wrapper. 

### Training Command
To start training the policy with 64 environments and Weights & Biases tracking enabled:
```bash
./isaaclab.sh -p ./scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Rizon4s-Forge-GearMesh-Direct-v0 --num_envs 64 --track --wandb-entity sianiemanuele_07 --wandb-name rizon

```

### Playing/Evaluating a Checkpoint

To visualize a trained checkpoint:

```bash
./isaaclab.sh -p ./scripts/reinforcement_learning/rl_games/play.py --checkpoint ./logs/rl_games/Forge/test/nn/rizon4s_200ep_512envs_increase_kp_scale.pth --task=Isaac-Rizon4s-Forge-GearMesh-Direct-v0 --num_envs 4

```

### Automatic Policy Export

The `rl_games` training script has been modified to automatically trace and export the final PyTorch model into a TorchScript (`.pt`) format for deployment.

The custom `export_rl_games_policy` function wraps the RL network and the running mean/std normalizer into a single `DeployWrapper`. It forces all weights and buffers to `float32` to prevent JIT type mismatch errors during real-world inference.

The exported policy and its corresponding YAML configuration will be saved automatically in:
`logs/rl_games/Forge/<run_name>/exported_policy/`

---

## 🚀 Running in Simulation

To run the deployment in simulation, you will need to open **three separate terminals** to handle the simulation engine, the ROS 2 message adapter, and the main deployment script.

Ensure that `SIMULATED = True` is set in `scripts/sim2real/run_assembly_task.py` before starting.

### Terminal 1: Isaac Sim & Wrench Publisher

This terminal initializes Isaac Sim and links its internal ROS 2 libraries.

```bash
# Activate Isaac Sim environment
conda activate env_isaaclab

# 1. CLEANUP: Ensure no system python paths interfere
unset PYTHONPATH
unset LD_LIBRARY_PATH

# 2. SETUP: Point to Isaac Sim's INTERNAL ROS libraries
# (Adjust the path if your conda env location is different)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/siani/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=42

# 3. Launch Isaac Sim
isaacsim

```

*Once Isaac Sim is open:*

1. Go to **Window → Script Editor**.
2. Paste the contents of `sim2real/utils/sim2sim_pub.py` into the editor.
3. Click **Run / Play** to start publishing the simulated wrench data.

### Terminal 2: ROS 2 Sim Adapter Node

This terminal runs the adapter that translates standard simulation messages into the official `flexiv_msgs` formats expected by the policy.

```bash
# Activate ROS/RL environment
conda activate ros_rl_env

# Source ROS workspace
source /opt/ros/humble/setup.bash
source ~/flexiv_ros2_ws/install/setup.bash

export ROS_DOMAIN_ID=42

# Navigate to the adapter script
cd /home/giuseppe/giuseppe_ws/repos/IsaacLab-thesis/scripts/sim2real/utils

# Run the adapter
python3 sim_adapter.py

```

### Terminal 3: Flexiv Bringup & Python Deploy Script

This terminal launches the fake hardware controllers and the main assembly policy.

```bash
# Activate ROS/RL environment
conda activate ros_rl_env

# Source ROS workspace
source /opt/ros/humble/setup.bash
source ~/flexiv_ros2_ws/install/setup.bash

export ROS_DOMAIN_ID=42

# 1. Navigate to deployment directory
cd /home/giuseppe/giuseppe_ws/repos/IsaacLab-thesis/scripts/sim2real 

# 2. Bring up the fake hardware controllers (run in background or separate tab)
ros2 launch flexiv_bringup rizon.launch.py robot_sn:=Rizon4-123456 rizon_type:=Rizon4s use_fake_hardware:=true load_gripper:=true &

# 3. Run the deployment script
python3 run_assembly_task.py

```

---

## 🦾 Running on the Real Robot

To run the pipeline on the physical Flexiv arm:

1. Open `run_assembly_task.py` and modify the configuration:
* Set `SIMULATED = False`
* Verify the `serial_number` matches your physical robot (e.g., `"Rizon4s-063126"`).


2. Connect to the Robot via a private Ethernet connection.
3. Launch the real robot hardware bringup (instead of `use_fake_hardware:=true`):

```bash
sudo su
source /opt/ros/jazzy/setup.bash
source /home/muletto-robot/ws/install/setup.bash 
export ROS_DOMAIN_ID=42
ros2 launch flexiv_bringup rizon.launch.py robot_sn:=Rizon4s-063126 rizon_type:=Rizon4s rdk_control_mode:=joint_impedance

```

4. Run `python3 run_assembly_task.py`. The script will wait for a 20-step tare calibration of the force/torque sensor before initiating the policy.

---

## 📊 Outputs and Logging

The control node automatically logs trajectory and force metrics. Upon success or termination, it will:

* Save episode statistics to `logs/sim2real_results.csv`.
* Generate and save matplotlib plots (Twists, Forces, Success Probability) inside the `logs/plots/` folder.

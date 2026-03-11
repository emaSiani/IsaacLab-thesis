# Isaac Lab Training Environments: Flexiv Rizon 4s Gear Assembly

This directory contains the reinforcement learning (RL) training environments for the Flexiv Rizon 4s robotic arm, built on top of NVIDIA Isaac Lab. The repository is divided into two primary environment configurations: **Factory** (base geometric/kinematic tasks) and **Forge** (contact-rich tasks with Force/Torque sensing designed for Sim2Real transfer).

## 📁 Directory Structure

The training source code is split into two main modules:

* `rizon4s_factory/`: The foundational environments.
  * Handles scene setup (spawning the robot, table, and assets).
  * Implements the core controller (`rizon4s_factory_control.py`) using Task-Space Impedance Control and Damped Least Squares (DLS) Inverse Kinematics.
  * Defines the base reward structure using a multi-scale keypoint approach (baseline, coarse, fine).
* `rizon4s_forge/`: The advanced Sim2Real environments. Inherits from `factory`.
  * **Force/Torque Sensing:** Adds simulated F/T sensor observations (`ft_force`).
  * **Success Prediction:** Adds an action dimension for the policy to predict task success, enabling early termination.
  * **Contact Penalties:** Introduces penalties for exceeding safe interaction forces.
  * **Advanced Randomization:** Uses `rizon4s_forge_events.py` to randomize rigid body mass, physics materials (friction, restitution), and sensor dead zones.

---

## 🛠️ Implemented Task

**Gear Mesh** (`Isaac-Rizon4s-Forge-GearMesh-Direct-v0`): Meshing a medium gear onto a base flanked by small and large gears. This task requires high-precision insertion and the dynamic alignment of three distinct sets of gear teeth.

> **Important Note:** While the codebase structure contains boilerplate and legacy files for other tasks (such as Peg Insert or Nut Thread), **the Gear Mesh task is the *only* one that has been actively modified, optimized, and trained** for this project.

### 🦾 Flexiv Rizon 4s Adaptation
The original FORGE environment natively supports the Franka Emika Panda. Adapting this for the Flexiv Rizon 4s required several critical modifications:
* **USD Replacement:** Integrating the official Rizon 4s USD with the GRAV gripper.
* **TCP Frame Alignment:** Adding a custom `fingertip_midpoint` reference frame to the Rizon 4s flange (offset by $0.19913$\,m) to match the expected FORGE coordinate system.
* **Gripper Kinematics:** Converting from the Panda's dual prismatic joints to the GRAV's complex revolute/mimic joint structure, requiring precise angular calibration ($0.122$\,rad) for a stable gear grasp.

---

## 🧠 State & Action Spaces (Forge)

### POMDP Formulation
The problem is formulated as a Partially Observable Markov Decision Process (POMDP). The agent uses an **Asymmetric Asynchronous Actor-Critic (A3C)** architecture. 

* **The Critic (Value Network)** receives the full ground-truth state $\mathcal{S}$ (poses, relative transforms, true velocities, joint configurations, and true contact forces).
* **The Actor (Policy Network)** receives only the noisy, partial observation vector $\Omega$ to ensure zero-shot Sim2Real transfer.

### Observation Space ($\Omega$)
The Actor receives a highly randomized, noise-injected vector:
* `fingertip_pos_rel_fixed`: Noisy 3D position of the EE relative to the target asset.
* `fingertip_quat`: Noisy 4D quaternion orientation of the EE.
* `ee_linvel` / `ee_angvel`: 6D linear and angular velocities (computed via finite differencing).
* `ft_force`: 3D Force readings from the simulated wrist sensor (smoothed with EMA and noise-injected).
* `force_threshold`: 1D contact penalty threshold for the current episode.
* `prev_actions`: The previous actions taken by the policy.

### Action Space
The policy outputs a **7-dimensional** action vector:
* `[0:3]`: Delta Position target (X, Y, Z). Scaled by $0.05$ and clipped to $\pm 2.0$\,cm per step.
* `[3:6]`: Delta Rotation target (Roll, Pitch, Yaw). Restricted to the Z-axis (Yaw) to align gear teeth, with a maximum rotation of $\pm 5.56^\circ$.
* `[6]`: Success prediction flag ([-1, 1] scaled to [0, 1]).

Actions are smoothed via an Exponential Moving Average (EMA) to prevent jittery behavior.

---

## 📏 Domain Randomization & Noise Injection

To ensure robust Sim2Real transfer, the environment utilizes extensive domain randomization and continuous noise injection.

### 1. Initial Spawn Randomization (Per Episode)
* **Fixed Asset Base:** Spawned within the absolute world coordinates $x \in [0.55, 0.65]$\,m, $y \in [-0.05, 0.05]$\,m, and $z \in [0.0, 0.1]$\,m. Its initial orientation is randomized by $\pm 15.0^\circ$ (Yaw).
* **Robot Hand (Relative to Base):** Position is randomized by $\pm 2.0$\,cm (X, Y) and orientation by $\pm 22.35^\circ$ (Yaw). Due to randomization, the initial displacement between the held asset and the fixed asset along the **Z-axis is strictly between 35 mm and 45 mm**.
* **Held Asset in Gripper:** Randomized by $\pm 3.0$\,mm (X, Z).
* **Physical Properties:** Held asset mass ($[0.0, 0.010]$\,kg) and fixed asset friction ($[0.25, 1.25]$).
* **Controller Dynamics:** EMA factor ($[0.025, 0.1]$), action scaling ($\pm 25\%$), rotation bounds ($\pm 29\%$), and proportional gains ($\pm 41\%$).

### 2. Maximum Exploration Boundary (Action Clipping)
The policy's Cartesian position targets are strictly clamped to a maximum displacement of **$\pm 5.0$\,cm** from the fixed asset. If the policy attempts to move beyond this, the target is aggressively clipped back to the 5\,cm boundary, forcing the agent to focus purely on the immediate assembly geometry.

### 3. Continuous Noise Injection (Per Step)
The trained agent always operates with noisy measurements to mirror real-world sensor non-idealities:
* **TCP Position:** Gaussian noise ($\sigma = 0.25$\,mm).
* **TCP Rotation:** Gaussian noise ($\sigma = 0.1^\circ$).
* **Force Sensor:** Smoothed via EMA and corrupted with Gaussian noise ($\sigma = 1.0$ Sim Units/N).

---

## 🎯 Reward Function Design

The reward $\mathcal{R}_t$ is a weighted sum designed to guide insertion while ensuring safety:

* **Keypoint Alignment ($r_{kp}$):** Uses a multi-stage squashing function (baseline, coarse, fine) to compute the distance between keypoints on the held and fixed assets.
* **Action Regularization ($r_{mag}, r_{grad}, r_{act, rel}$):** Penalizes high-effort control signals, temporal oscillations, and large displacements relative to the asset frame.
* **Success Prediction Error ($r_{pred}$):** Penalizes the absolute error between the policy's success prediction and the ground-truth success (activated after a 25% success ratio is reached).
* **Force-Thresholding ($r_{force}$):** A crucial safety feature that applies a ReLU penalty if smoothed forces exceed a randomized threshold ($\tau_f \in [5, 10]$).
* **Discrete Bonuses:** Awarded for engagement and final success.

---

## 🚀 Training and Evaluation

Training is managed via `rl_games` using a **Proximal Policy Optimization (PPO)** algorithm with an LSTM network (1024 hidden units) to encode temporal context. We use 512 parallel environments and an adaptive learning rate scheduler based on the Kullback-Leibler (KL) Divergence (threshold $0.008$).

### Start Training
Run the `isaaclab.sh` executable, specifying the python training script and the Gear Mesh task. 

```bash
# Train the Gear Mesh task with 512 parallel environments
./isaaclab.sh -p ./scripts/reinforcement_learning/rl_games/train.py \
  --task=Isaac-Rizon4s-Forge-GearMesh-Direct-v0 \
  --num_envs 512 \
  --track \
  --wandb-entity <your_wandb_entity> \
  --wandb-name rizon_gear_mesh

```

### Play/Evaluate a Policy

To visualize a trained policy, use the `play.py` script and pass the path to your `.pth` checkpoint.

```bash
# Evaluate a trained Gear Mesh checkpoint
./isaaclab.sh -p ./scripts/reinforcement_learning/rl_games/play.py \
  --checkpoint ./logs/rl_games/Forge/rizon_gear_mesh/nn/last_model.pth \
  --task=Isaac-Rizon4s-Forge-GearMesh-Direct-v0 \
  --num_envs 4

```

> **Note:** The `rl_games` training script has been customized to automatically export the final policy into a TorchScript (`.pt`) file alongside a `_env_config.yaml` file, ready to be deployed on the physical robot.

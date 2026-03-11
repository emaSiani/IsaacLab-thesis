# Isaac Lab Snapshot: Flexiv Rizon 4s End-to-End Sim2Real Pipeline

This archive contains a complete, customized snapshot of **NVIDIA Isaac Lab**. It has been specifically modified and extended to provide an end-to-end Reinforcement Learning (RL) pipeline—from training in simulation to physical deployment—for the Flexiv Rizon 4s robotic arm, with a primary focus on contact-rich assembly tasks like Gear Meshing.

By packaging the entire Isaac Lab framework, this snapshot ensures perfect version compatibility between the underlying physics engine, the customized training environments, and the deployment utilities.

---

## 🌟 Key Additions & Modifications

This snapshot introduces three major architectural contributions on top of the base Isaac Lab framework:

1. **Custom Training Environments (Factory & Forge)**
   * Located in `source/isaaclab_tasks/isaaclab_tasks/direct/`.
   * We have adapted the NVIDIA FORGE and Factory environments (originally designed for the Franka Emika Panda) to fully support the **Flexiv Rizon 4s**. 
   * Includes custom USD assets, re-aligned Tool Center Point (TCP) frames, tailored initialization logic for the GRAV gripper's mimic joints, and specific domain randomizations to close the Sim-to-Real gap.

2. **Automated Policy Export (`rl_games`)**
   * Located in `scripts/reinforcement_learning/rl_games/`.
   * The standard training loop has been patched. Upon training completion (or during evaluation), the framework now automatically wraps the neural network and normalizer, forces `float32` types, and traces the model into a deployable **TorchScript (`.pt`) format**.

3. **Sim2Real Deployment Pipeline**
   * Located in `scripts/sim2real/`.
   * A complete ROS 2-based deployment architecture capable of controlling both the simulated and the real physical robot. It features a custom node to bridge simulated PhysX wrenches to ROS, Pinocchio-based Differential Inverse Kinematics, and real-time execution of the exported PyTorch policies.

---

## 📁 Repository Highlights (Where to find what)

To navigate this large snapshot, here are the critical directories that have been added or modified:

```text
IsaacLab-Snapshot/
├── scripts/
│   ├── sim2real/                 # [NEW] Deployment framework (run_assembly_task.py, ROS adapters)
│   │   └── requirements/         # [NEW] Conda env files (env_isaaclab.yml, ros_rl_env.yml)
│   └── reinforcement_learning/   # [MODIFIED] rl_games with auto-export to TorchScript
├── source/
│   └── isaaclab_tasks/isaaclab_tasks/direct/
│       ├── rizon4s_factory/      # [NEW] Base kinematic assembly environments
│       └── rizon4s_forge/        # [NEW] Advanced force-aware Sim2Real environments
└── robots/                       # [NEW] Flexiv Rizon 4s USD assets

```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Before running any scripts, ensure you have created the necessary Conda environments using the provided requirement files located in the deployment folder:

```bash
conda env create -f scripts/sim2real/requirements/env_isaaclab.yml
conda env create -f scripts/sim2real/requirements/ros_rl_env.yml

```

### 2. Training the Policy

To train the Gear Mesh policy using the customized FORGE environment:

```bash
conda activate env_isaaclab
./isaaclab.sh -p ./scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Rizon4s-Forge-GearMesh-Direct-v0 --num_envs 512

```

*The trained `.pt` policy and its `_env_config.yaml` will be automatically exported to the `logs/` directory.*

### 3. Deployment (Sim2Real)

Copy your exported `.pt` policy into the `scripts/sim2real/robots/rizon/policies/` folder.
To run the deployment, you will use the `ros_rl_env` and orchestrate the simulation bridge and the control node.

---

## 📖 Further Documentation (Detailed READMEs)

To keep this root document concise, deep-dives into the specific components are split into their own dedicated README files. Please consult the following documents located within the snapshot:

* **Training Architecture:** Located at `source/isaaclab_tasks/isaaclab_tasks/direct/README.md`.
* *Read this for:* Details on the POMDP formulation, reward structure, continuous noise injection, and the exact domain randomization bounds.


* **Deployment Architecture:** Located at `scripts/sim2real/README.md`.
* *Read this for:* Full, step-by-step terminal commands regarding both Simulated and Real-World deployment, the ROS 2 node setup, the Isaac Sim physics adapter, and Pinocchio DLS IK logic.

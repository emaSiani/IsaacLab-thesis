# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import matplotlib.pyplot as plt # [NEW] For plotting
import numpy as np              # [NEW] For data handling

from isaaclab.app import AppLauncher
from policy_export import export_rl_games_policy 


# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import random
import time
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()
    dt = env.unwrapped.step_dt

    # 1. Esporta la policy in ONNX e TorchScript (.pt)
    # export_rl_games_policy(agent, log_dir, task_name, rl_device)
        
    # ----------------------------------------------------------------------
    # Fine Codice di Esportazione
    # ----------------------------------------------------------------------
    
    # reset environment
    obs = env.reset()

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()

    # --- [NEW] FORENSIC LOGGING SETUP ---
    print("\n🔍 STARTING FORENSIC LOGGING (Sim-to-Sim)...")
    log_data = {
        "steps": [],
        "actions": [],      # [vx, vy, vz, wx, wy, wz]
        "forces": [],       # [fx, fy, fz] from Observation
        "success_pred": [], # From Action[6]
        "rewards": []
    }
    # Observation Indices from your Config
    IDX_FORCE_START = 13
    IDX_FORCE_END = 16
    MAX_STEPS = 400 
    step_counter = 0
    # ------------------------------------

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, rew, dones, _ = env.step(actions)

            # --- [FIXED] CAPTURE FORENSIC DATA ---
            # We use Environment 0 for plotting
            
            # 1. Handle Observation Dictionary
            if isinstance(obs, dict):
                # Try common keys used in Isaac Lab / RL Games
                if "policy" in obs:
                    raw_obs_tensor = obs["policy"]
                elif "obs" in obs:
                    raw_obs_tensor = obs["obs"]
                else:
                    # Fallback: grab the first value available
                    raw_obs_tensor = next(iter(obs.values()))
            else:
                raw_obs_tensor = obs

            # 2. Slice the first environment
            current_obs = raw_obs_tensor[0].detach().cpu().numpy()
            
            # 3. Actions and Rewards (Usually tensors)
            current_action = actions[0].detach().cpu().numpy()
            current_rew = rew[0].item()

            log_data["steps"].append(step_counter)

            # 1. Scale Actions (To match Deployment Plots)
            scaled_action = current_action.copy()
            scaled_action[0:3] *= 0.05  # Pos Scale from assembly.py
            scaled_action[3:6] *= 1.0   # Rot Scale from assembly.py
            log_data["actions"].append(scaled_action[:6]) 

            # 2. Success Prediction
            raw_pred = current_action[6]
            prob_pred = (raw_pred + 1.0) / 2.0
            log_data["success_pred"].append(prob_pred)

            # 3. Forces (Observed)
            force_obs = current_obs[IDX_FORCE_START:IDX_FORCE_END]
            log_data["forces"].append(force_obs)

            log_data["rewards"].append(current_rew)
            step_counter += 1

            if dones[0] or step_counter >= MAX_STEPS:
                print(f"🛑 Episode ended at step {step_counter}. Generating Forensic Plot...")
                break
            # -----------------------------------

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # --- [NEW] PLOTTING LOGIC ---
    save_dir = "logs/play_forensics"
    os.makedirs(save_dir, exist_ok=True)

    steps = np.array(log_data["steps"])
    actions = np.array(log_data["actions"])
    forces = np.array(log_data["forces"])
    success = np.array(log_data["success_pred"])

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Actions (Twist)
    axs[0].plot(steps, actions[:, 0], label='Vx (Rel)', color='r', linestyle='-')
    axs[0].plot(steps, actions[:, 1], label='Vy (Rel)', color='g', linestyle='-')
    axs[0].plot(steps, actions[:, 2], label='Vz (Rel)', color='b', linestyle='-')
    axs[0].set_title(f"1. Policy Actions (Twist) - {task_name}")
    axs[0].set_ylabel("Scaled Action (m/s)")
    axs[0].legend(ncol=3, fontsize='small')
    axs[0].grid(True, alpha=0.3)

    # Plot 2: Forces (What the Policy SEES)
    axs[1].plot(steps, forces[:, 0], label='Fx (Obs)', color='r', alpha=0.7)
    axs[1].plot(steps, forces[:, 1], label='Fy (Obs)', color='g', alpha=0.7)
    axs[1].plot(steps, forces[:, 2], label='Fz (Obs)', color='b', alpha=0.7)
    axs[1].set_title("2. Forces Observed by Policy")
    axs[1].set_ylabel("Force (N)")
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)

    # Plot 3: Success Prediction
    axs[2].plot(steps, success, label='Success Prob', color='purple', linewidth=2)
    axs[2].axhline(y=0.9, color='k', linestyle='--', label='Threshold')
    axs[2].set_title("3. Policy Self-Confidence")
    axs[2].set_ylabel("Probability [0-1]")
    axs[2].set_xlabel("Sim Steps")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "sim_forensics.png")
    plt.savefig(plot_path)
    print(f"🖼️ FORENSIC PLOT SAVED: {plot_path}")
    # ----------------------------

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
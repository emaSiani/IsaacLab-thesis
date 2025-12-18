# export_env.py
# Usage: python export_env.py --task=Isaac-Your-Task-Name

import argparse
import sys
import os
import time
from isaaclab.app import AppLauncher

# 1. Parse Args
parser = argparse.ArgumentParser(description="Export Isaac Lab Env to USD (Fixed Positions).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--filename", type=str, default="deployment_env.usd", help="Output filename.")
parser.add_argument("--settle_steps", type=int, default=100, help="Steps to run before saving.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
import omni.timeline
import gymnasium as gym
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

# =============================================================================
# HELPER: FORCE USD WRITEBACK (ROBUST)
# =============================================================================
def force_usd_writeback():
    """Forces the Physics Engine to update the USD Stage every frame."""
    print("[INFO] Forcing Physics -> USD Writeback...")
    stage = omni.usd.get_context().get_stage()
    
    # Scan stage for any prim with type "PhysicsScene"
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsScene":
            # Attributes that force the sim to write positions back to the file
            attrs_to_set = [
                "physxScene:updatePositionsToUsd", 
                "physxScene:updateVelocitiesToUsd"
            ]
            for attr_name in attrs_to_set:
                attr = prim.GetAttribute(attr_name)
                if attr.IsValid():
                    attr.Set(True)
    
# =============================================================================
# MAIN
# =============================================================================
@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point") 
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    
    # 1. SETUP
    env_cfg.scene.num_envs = 1
    # Disable Fabric so the simulation actually runs on the USD stage
    if hasattr(env_cfg, "sim"):
        env_cfg.sim.use_fabric = False

    # 2. CREATE ENV
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # 3. RESET (Triggers Randomization)
    print("[INFO] Resetting environment (Triggering Randomization)...")
    env.reset()
    
    # 4. FORCE WRITEBACK
    # Crucial: Ensures the randomized positions are written to the stage
    force_usd_writeback()

    # 5. SETTLE LOOP
    # We run physics for X steps to let objects move to their randomized targets
    print(f"[INFO] Stepping physics for {args_cli.settle_steps} frames to settle objects...")
    for i in range(args_cli.settle_steps):
        env.unwrapped.sim.step()
        if i % 10 == 0:
            print(f"   ... step {i}/{args_cli.settle_steps}")

    # 6. PAUSE TIMELINE
    # Stop the timeline so we capture the exact current frame
    omni.timeline.get_timeline_interface().pause()

    # 7. SAVE
    output_path = os.path.abspath(args_cli.filename)
    print(f"\n[INFO] Saving Randomized Stage to: {output_path}")
    omni.usd.get_context().save_as_stage(output_path, None)
    
    print("[SUCCESS] Export complete. Open this file in Isaac Sim.")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run IsaacLab Simulation.
Mode: NATIVE GRAPH + ZOMBIE ENV (Robust)
Fix: Corrected Monkey Patch Signature
"""

import argparse
import sys
import os
import torch
import types

from isaaclab.app import AppLauncher

# 1. Parse Args
parser = argparse.ArgumentParser(description="Run IsaacLab Simulation via Native Bridge.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Ignored. Forced to 1.") 
parser.add_argument("--robot_path", type=str, default=None, help="USD Prim path of the robot.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app
import omni.timeline
import omni.graph.core as og
import omni.usd
from pxr import Usd, UsdPhysics
import gymnasium as gym
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

# Enable Extensions
manager = omni.kit.app.get_app().get_extension_manager()
exts = ["isaacsim.ros2.bridge"]
for ext in exts:
    if not manager.is_extension_enabled(ext):
        manager.set_extension_enabled_immediate(ext, True)

# =============================================================================
# FIND ROBOT
# =============================================================================
def find_robot_prim_path():
    stage = omni.usd.get_context().get_stage()
    print("\n" + "="*30)
    print(" [DEBUG] SCANNING STAGE FOR ROBOT")
    print("="*30)
    
    found_path = None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            found_path = str(prim.GetPath())
            print(f" [FOUND] Articulation Root: {found_path}")
            break
            
    if not found_path:
        found_path = "/World/envs/env_0/Robot"
        print(f" [WARN] No root API found. Defaulting to: {found_path}")

    return found_path

# =============================================================================
# ACTION GRAPH
# =============================================================================
def create_disconnected_graph(robot_prim_path):
    keys = og.Controller.Keys
    graph_path = "/ActionGraph"
    
    try:
        if og.Controller.graph_exists(graph_path):
            og.Controller.delete_node(graph_path)
    except: pass

    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),

                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [
                ("ArticulationController.inputs:robotPath", robot_prim_path),
                ("PublishJointState.inputs:targetPrim", robot_prim_path),
                ("PublishTF.inputs:targetPrims", [robot_prim_path]),
                ("SubscribeJointState.inputs:topicName", "/isaac_joint_commands"),
                ("PublishJointState.inputs:topicName", "/joint_states"),
                ("PublishClock.inputs:topicName", "/clock"),
            ],
        },
    )
    print(f"[INFO] Graph Built (DISCONNECTED).")

def connect_bridge():
    keys = og.Controller.Keys
    graph_path = "/ActionGraph"
    print("[INFO] Connecting ROS Bridge...")
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CONNECT: [
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ],
        },
    )
    print("[INFO] Bridge Connected!")

# =============================================================================
# MAIN
# =============================================================================
@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point") 
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = 1
    if args_cli.disable_fabric and hasattr(env_cfg, "sim"):
        env_cfg.sim.use_fabric = False

    env = gym.make(args_cli.task, cfg=env_cfg)

    # --- ZOMBIE ENV PATCH (FIXED SIGNATURES) ---
    
    # 1. Disable Control
    # Fixed: Removed 'action' arg. Matches internal signature: _apply_action(self)
    def no_op_apply_action(self): 
        pass
    env.unwrapped._apply_action = types.MethodType(no_op_apply_action, env.unwrapped)

    # 2. Disable Rewards
    # Fixed: Returns zero tensor to prevent downstream crashes
    def no_op_get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)
    env.unwrapped._get_rewards = types.MethodType(no_op_get_rewards, env.unwrapped)
    # -------------------------------------------

    robot_prim_path = find_robot_prim_path()
    create_disconnected_graph(robot_prim_path)

    env.reset()
    omni.timeline.get_timeline_interface().play()
    
    print("\n" + "="*60)
    print(" SIMULATION READY (Passive Mode)")
    print(" 1. Run your 'run_assembly_task.py' NOW.")
    print(" 2. Press ENTER here to connect.")
    print("="*60 + "\n")

    # Safe Action Tensor
    try:
        if hasattr(env.unwrapped.action_space, 'shape'):
            num = env.unwrapped.action_space.shape[0] if len(env.unwrapped.action_space.shape) == 1 else env.unwrapped.action_space.shape[1]
        else:
            num = 7
    except: num = 7
        
    zero_action = torch.zeros((env.unwrapped.num_envs, num), device=env.unwrapped.device)

    # Warmup
    for _ in range(50):
        with torch.inference_mode():
            env.step(zero_action)

    input(">>> PRESS ENTER TO CONNECT ROS BRIDGE <<<")
    connect_bridge()

    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(zero_action)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
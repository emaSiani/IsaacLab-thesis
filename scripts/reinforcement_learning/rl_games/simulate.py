# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run IsaacLab Simulation.
Mode: NATIVE GRAPH (JointState)
Fix: Removed crashing ROS2Context node
"""

import argparse
import sys
import os

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

    print("="*30 + "\n")
    return found_path

# =============================================================================
# ACTION GRAPH
# =============================================================================
def setup_graph(robot_prim_path):
    keys = og.Controller.Keys
    graph_path = "/ActionGraph"
    
    try:
        if og.Controller.graph_exists(graph_path):
            og.Controller.delete_node(graph_path)
    except: pass

    try:
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    
                    # --- NODES (No ROS2Context) ---
                    ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                keys.CONNECT: [
                    # Execution
                    ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),

                    # Data
                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    
                    # Time
                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ],
                keys.SET_VALUES: [
                    ("ArticulationController.inputs:robotPath", robot_prim_path),
                    ("PublishJointState.inputs:targetPrim", robot_prim_path),
                    ("PublishTF.inputs:targetPrims", [robot_prim_path]),
                    
                    # --- TOPICS ---
                    # Listens to what your Brain publishes (JointState)
                    ("SubscribeJointState.inputs:topicName", "/isaac_joint_commands"),
                    
                    ("PublishJointState.inputs:topicName", "/joint_states"),
                    ("PublishClock.inputs:topicName", "/clock"),
                ],
            },
        )
        print(f"[INFO] >>> GRAPH CREATED SUCCESSFULLY <<<")
    except Exception as e:
        print(f"[ERROR] Graph Creation Failed: {e}")

# =============================================================================
# MAIN
# =============================================================================
@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point") 
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = 1
    if args_cli.disable_fabric and hasattr(env_cfg, "sim"):
        env_cfg.sim.use_fabric = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot_prim_path = find_robot_prim_path()
    setup_graph(robot_prim_path)

    env.reset()
    omni.timeline.get_timeline_interface().play()
    
    print("\n" + "="*50)
    print(" SIMULATION RUNNING")
    print(f" ROS_DOMAIN_ID: {os.environ.get('ROS_DOMAIN_ID', 0)}")
    print(" Listening to: /isaac_joint_commands (JointState)")
    print("="*50 + "\n")

    while simulation_app.is_running():
        simulation_app.update()

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
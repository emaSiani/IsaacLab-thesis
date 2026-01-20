# FILE: isaac_sender.py
import socket
import json
import torch
import os
import argparse
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot_path", type=str, default="/World/Rizon4s_with_Grav")
parser.add_argument("--sensor_link", type=str, default="flange") 
parser.add_argument("--usd_path", type=str, default="/home/giuseppe/giuseppe_ws/repos/IsaacLab-thesis/source/robots/deployment_env_fingertip.usd")
parser.add_argument("--no-v-sync", action="store_true", default=True) 
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation, ArticulationCfg
import isaacsim.core.utils.torch as torch_utils
from isaaclab.actuators import ImplicitActuatorCfg

LISTEN_IP = "0.0.0.0"       
TARGET_IP = "127.0.0.1"     
UDP_PORT_SEND = 50001 
UDP_PORT_RECV = 50002 

DEBUG = True

sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_recv.bind((LISTEN_IP, UDP_PORT_RECV))
sock_recv.setblocking(False)
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def main():
    if not os.path.exists(args_cli.usd_path): return
    omni.usd.get_context().open_stage(args_cli.usd_path)

    sim = SimulationContext(cfg=SimulationCfg(dt=0.005, device="cuda:0"))
    sim.set_camera_view([1.2, 0.0, 1.2], [0.0, 0.0, 0.0])

    robot_cfg = ArticulationCfg(
        prim_path=args_cli.robot_path,
        spawn=None, 
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint1": -0.13, "joint2": -0.5707, "joint3": 0.422, "joint4": 1.7854, "joint5": -0.317, "joint6": 0.705, "joint7": 0.0}),
        actuators={"arm": ImplicitActuatorCfg(joint_names_expr=["joint.*"], stiffness=800.0, damping=40.0)}
    )
    robot = Articulation(cfg=robot_cfg)
    sim.reset()
    
    try: flange_idx = robot.body_names.index(args_cli.sensor_link)
    except: flange_idx = -1

    target_joint_pos = robot.data.default_joint_pos.clone()
    smooth_wrench = torch.zeros((1, 6), device=sim.device)
    alpha = 0.1 
    RENDER_FPS = 30
    render_interval = int(6.6) # approx 30fps at 200hz
    step_counter = 0

    while simulation_app.is_running():
        try:
            while True:
                data, _ = sock_recv.recvfrom(1024)
                cmd_list = json.loads(data.decode('utf-8'))
                if DEBUG:
                    print(f"✅ Ricevuto Comando: {cmd_list}")
                if len(cmd_list) >= 7:
                    target_joint_pos[:, 0:7] = torch.tensor(cmd_list[:7], device=sim.device)
        except: pass

        robot.set_joint_position_target(target_joint_pos)
        sim.step(render=(step_counter % render_interval == 0))
        robot.update(0.005)

        # Sensors
        tcp_pos_w = robot.data.body_pos_w[:, flange_idx].view(-1, 3)
        tcp_quat_w = robot.data.body_quat_w[:, flange_idx].view(-1, 4)
        curr_joint_pos = robot.data.joint_pos[:, 0:7]
        curr_joint_vel = robot.data.joint_vel[:, 0:7]

        raw_wrench_body = robot.root_physx_view.get_link_incoming_joint_force()[:, flange_idx, :].view(-1, 6)
        f_world = torch_utils.quat_apply(tcp_quat_w, raw_wrench_body[:, 0:3])
        t_world = torch_utils.quat_apply(tcp_quat_w, raw_wrench_body[:, 3:6])
        current_wrench_world = torch.cat([f_world, t_world], dim=1)
        smooth_wrench = alpha * current_wrench_world + (1 - alpha) * smooth_wrench
        
        payload = {
            "tcp_pos": tcp_pos_w[0].cpu().tolist(),
            "tcp_quat": tcp_quat_w[0].cpu().tolist(),
            "wrench": smooth_wrench[0].cpu().tolist(),
            "joint_pos": curr_joint_pos[0].cpu().tolist(),
            "joint_vel": curr_joint_vel[0].cpu().tolist()
        }
        
        try: sock_send.sendto(json.dumps(payload).encode('utf-8'), (TARGET_IP, UDP_PORT_SEND))
        except: pass
        step_counter += 1

if __name__ == "__main__":
    main()
    simulation_app.close()
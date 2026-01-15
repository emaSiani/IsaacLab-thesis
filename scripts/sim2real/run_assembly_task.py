# FILE: scripts/sim2real/run_assembly_task.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
import numpy as np
from scipy.spatial.transform import Rotation as R
from flexiv_msgs.msg import RobotStates
from flexiv_msgs.action import Move
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from robots.rizon.assembly import FlexivGearAssemblyPolicy
from utils.angle_utils import map_joint_angle

SIMULATED = True
DEBUG = False

class FlexivAssemblyNode(Node):
    def __init__(self):
        super().__init__("flexiv_assembly_node")
        
        # --- CONFIG ---
        self.SERIAL = "" # <--- CHECK SN
        self.GEAR_OFFSET_Z = 0.19913 # Distance from Flange to Gear Tip
        self.step_counter = 0

        # State Machine
        self.STATE_INIT_GRASP = 0
        self.STATE_RUNNING = 1

        # self.STATE_INIT_GRASP if the robot needs to grasp the gear
        self.current_mode = self.STATE_RUNNING
        self.grasp_timer = 0

        # Policy
        self.policy = FlexivGearAssemblyPolicy()
        
        # IO
        self.sub_states = self.create_subscription(RobotStates, f"/{self.SERIAL.replace('-', '_')}/flexiv_robot_states", self.cb_states, 1) if self.SERIAL else self.create_subscription(RobotStates, "/flexiv_robot_states", self.cb_states, 1)
        print(f'Created subscriber on topic /{self.SERIAL.replace("-", "_")}/flexiv_robot_states' if self.SERIAL else 'Created subscriber on topic /flexiv_robot_states')
        self.pub_arm = self.create_publisher(JointTrajectory, "/rizon_arm_controller/joint_trajectory", 1)
        print('Created publisher on topic /rizon_arm_controller/joint_trajectory')
        # self.client_gripper = ActionClient(self, Move, f"/{self.SERIAL}/tool/move") if self.SERIAL else ActionClient(self, Move, "/tool/move")

        self.robot_state = None
        self.create_timer(0.02, self.control_loop)
        self.get_logger().info("Node Started. Mode: INITIALIZING GRASP")

    def cb_states(self, msg):
        self.robot_state = msg
        if DEBUG and self.step_counter % 50 == 0:
            self.get_logger().info(f'Received Robot State Message: {msg}')

    def control_loop(self):
        if self.robot_state is None: return

        # 1. Parse State (Flange Pose)
        flange_pos = np.array([
            self.robot_state.tcp_pose.pose.position.x,
            self.robot_state.tcp_pose.pose.position.y,
            self.robot_state.tcp_pose.pose.position.z
        ])
        # Isaac expects [w, x, y, z]
        flange_quat = np.array([
            self.robot_state.tcp_pose.pose.orientation.w,
            self.robot_state.tcp_pose.pose.orientation.x,
            self.robot_state.tcp_pose.pose.orientation.y,
            self.robot_state.tcp_pose.pose.orientation.z
        ])

        wrench = np.array([
            self.robot_state.ext_wrench_in_world.wrench.force.x,
            self.robot_state.ext_wrench_in_world.wrench.force.y,
            self.robot_state.ext_wrench_in_world.wrench.force.z
        ])
        
        # 2. Apply Gear Offset (Flange -> Gear Tip)
        # Convert Quat to Rotation Matrix
        # Scipy uses [x, y, z, w]
        # r = R.from_quat([flange_quat[1], flange_quat[2], flange_quat[3], flange_quat[0]])
        # offset_world = r.apply([0.0, 0.0, self.GEAR_OFFSET_Z])
        
#         gear_pos = flange_pos + offset_world
        gear_pos = flange_pos - [0.0, 0.0, self.GEAR_OFFSET_Z]
        gear_quat = flange_quat # Orientation is same, just translated

        # print robots' state every 1 second if debug is true
        if DEBUG and self.robot_state is not None and self.step_counter % 50 == 0:
            self.get_logger().info(f"Step: {self.step_counter}")
            self.get_logger().info(f"Gear Pos: {gear_pos}")
            self.get_logger().info(f"Gear Quat: {gear_quat}")
            self.get_logger().info(f"Wrench: {wrench}")

        # --- STATE MACHINE ---
        
        if self.current_mode == self.STATE_INIT_GRASP:
            # Send Grasp Command ONCE (force closure)
            if self.grasp_timer == 0:
                self.send_gripper(0.0) # Close
                self.get_logger().info("Closing Gripper...")
            
            self.grasp_timer += 1
            # Wait 2 seconds (50Hz * 2s = 100 ticks) for grasp to settle
            if self.grasp_timer > 100:
                self.current_mode = self.STATE_RUNNING
                self.get_logger().info("Grasp Complete. STARTING POLICY.")
            return

        elif self.current_mode == self.STATE_RUNNING:
            # 3. Run Policy
            arm_cmd, _ = self.policy.compute_action(gear_pos, gear_quat, wrench)

            # 4. Publish
            traj = JointTrajectory()
            traj.header.stamp = self.get_clock().now().to_msg()
            traj.joint_names = [f"{self.SERIAL}_joint{i}" for i in range(1, 8)] if self.SERIAL else [f"joint{i}" for i in range(1, 8)]
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in arm_cmd]
            pt.time_from_start = Duration(seconds=0.02).to_msg()
            traj.points.append(pt)
            self.pub_arm.publish(traj)
            
            # Enforce Closed Gripper
            # (Optional: send periodically if needed, but usually one close is enough)
            # self.send_gripper(0.0)
        self.step_counter += 1

    def send_gripper(self, width):
        goal = Move.Goal()
        goal.width = width
        goal.velocity = 0.1
        goal.max_force = 40.0 # Strong grasp
        self.client_gripper.send_goal_async(goal)

def main():
    rclpy.init()
    node = FlexivAssemblyNode()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
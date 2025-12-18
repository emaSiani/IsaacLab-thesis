import math
import time
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
from tf2_ros import TransformException
import tf_transformations as tr

# Import Policy Logic
from robots.rizon.assembly import FlexivGearAssemblyPolicy
from utils.angle_utils import map_joint_angle

class FlexivAssemblyNode(Node):
    """ROS2 node for deploying Gear Assembly on Flexiv Rizon 4s."""

    PI = math.pi
    # Simulation/Mapping limits
    ARM_SIM_DOF_ANGLE_LIMITS = [(-360, 360, False)] * 7
    # Real Servo limits
    ARM_SERVO_ANGLE_LIMITS = [(-2 * PI, 2 * PI)] * 7
    
    JOINT_STATE_TOPIC = "/joint_states"
    CMD_TOPIC = "/rizon_arm_controller/joint_trajectory"
    
    # N.B: Joint names could include robot's serial number
    # serial_number="Rizon4s-063126"
    serial_number=""
    FLEXIV_JOINT_NAMES = [
        f"{serial_number}_joint1",
        f"{serial_number}_joint2",
        f"{serial_number}_joint3",
        f"{serial_number}_joint4",
        f"{serial_number}_joint5",
        f"{serial_number}_joint6",
        f"{serial_number}_joint7"
    ] if serial_number else [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7"
    ]

    BASE_FRAME = f"{serial_number}_base_link" if serial_number else "base_link"
    FLANGE_FRAME = f"{serial_number}_flange" if serial_number else "flange"

    # =========================================================================

    def __init__(self):
        super().__init__("flexiv_assembly_node")

        # Initialize Policy
        try:
            self.robot = FlexivGearAssemblyPolicy()
            self.get_logger().info("Policy loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize policy: {e}")
            raise e

        # Control Loop
        self.control_freq = 50.0  
        self.step_size = 1.0 / self.control_freq 
        self.timer = self.create_timer(self.step_size, self.step_callback)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers & Publishers
        self.sub_joints = self.create_subscription(
            JointState, self.JOINT_STATE_TOPIC, self.joint_state_callback, 1
        )
        self.pub_cmd = self.create_publisher(JointTrajectory, self.CMD_TOPIC, 1)

        # Offset Gear (Distance Flange -> Gear Tip)
        # Adjust this based on your gripper + gear length
        self.GEAR_TIP_OFFSET_Z = 0.19913

        self.flag_target_reached = False
        
        self.joint_map = {n: 0.0 for n in self.FLEXIV_JOINT_NAMES}
        self.vel_map = {n: 0.0 for n in self.FLEXIV_JOINT_NAMES}
        
        self.get_logger().info(f"Node initialized. Listening for joints: {self.FLEXIV_JOINT_NAMES[0]}...")

    

    def joint_state_callback(self, msg: JointState):
        """Updates joint states. Filters out non-arm joints (like gripper fingers)."""
        found_any = False
        for i, name in enumerate(msg.name):
            if name in self.joint_map:
                self.joint_map[name] = msg.position[i]
                if len(msg.velocity) > i:
                    self.vel_map[name] = msg.velocity[i]
                found_any = True
        
        # Only update policy if we actually found arm joints in this message
        if found_any:
            ordered_pos = [self.joint_map[n] for n in self.FLEXIV_JOINT_NAMES]
            ordered_vel = [self.vel_map[n] for n in self.FLEXIV_JOINT_NAMES]
            self.robot.update_joint_state(ordered_pos, ordered_vel)

    def get_gear_tip_position(self):
        """Calculates Gear Tip position using TF + Offset."""
        try:
            # Transform Base -> Flange
            t = self.tf_buffer.lookup_transform(self.BASE_FRAME, self.FLANGE_FRAME, rclpy.time.Time())
            
            q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            rot_matrix = tr.quaternion_matrix(q)
            
            # Local offset along Flange Z
            offset_vector = np.array([0.0, 0.0, self.GEAR_TIP_OFFSET_Z, 1.0])
            world_offset = np.dot(rot_matrix, offset_vector)
            
            flange_pos = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
            
            return flange_pos + world_offset[:3]
        except TransformException as ex:
            self.get_logger().warn(f"TF Lookup Failed: {ex}")
            return None
    def joint_state_callback(self, msg: JointState):
        """Updates joint states. Filters out non-arm joints."""
        found_any = False
        
        if not hasattr(self, "_debug_first_joint_msg"):
            self.get_logger().info(f"[DEBUG] Received JointState names: {msg.name}")
            self._debug_first_joint_msg = True

        for i, name in enumerate(msg.name):
            if name in self.joint_map:
                self.joint_map[name] = msg.position[i]
                if len(msg.velocity) > i:
                    self.vel_map[name] = msg.velocity[i]
                found_any = True
        
        if found_any:
            ordered_pos = [self.joint_map[n] for n in self.FLEXIV_JOINT_NAMES]
            ordered_vel = [self.vel_map[n] for n in self.FLEXIV_JOINT_NAMES]
            self.robot.update_joint_state(ordered_pos, ordered_vel)
        else:
            # if here, then we are not finding the joint names
            self.get_logger().warn(f"Joint names are incorrect. Received {self.joint_map.keys} while looking for {self.FLEXIV_JOINT_NAMES}")
            pass 

    def check_target_reached(self, current_tip_pos):
        target_pos = self.robot.peg_target_pose[:3]
        dist = np.linalg.norm(current_tip_pos - target_pos)
        return dist < 0.005 # 5mm threshold

    def step_callback(self):
        if self.flag_target_reached:
            return

        # CHECK 1: TF
        gear_tip_pos = self.get_gear_tip_position()
        if gear_tip_pos is None: 
            return
            
        self.robot.update_gear_tip_position(gear_tip_pos)

        joint_cmd = self.robot.forward(self.step_size)
        if joint_cmd is None:
            self.get_logger().warn("[DEBUG POLICY FAIL] Policy returned None. Waiting for joint states...", throttle_duration_sec=2.0)
            return

        # Se arrivi qui, stiamo calcolando e pubblicando
        self.get_logger().info(f"[DEBUG RUNNING] Action calculated. Target Joint 1: {joint_cmd[0]:.3f}", throttle_duration_sec=1.0)

        traj = JointTrajectory()
        traj.joint_names = self.FLEXIV_JOINT_NAMES
        point = JointTrajectoryPoint()
        
        target_pos = []
        for i, val in enumerate(joint_cmd):
            target_pos.append(map_joint_angle(val, i, self.ARM_SIM_DOF_ANGLE_LIMITS, self.ARM_SERVO_ANGLE_LIMITS))
        
        point.positions = target_pos
        point.time_from_start = Duration(sec=0, nanosec=int(self.step_size * 1e9))
        traj.points.append(point)
        self.pub_cmd.publish(traj)

        if self.check_target_reached(gear_tip_pos):
            self.get_logger().info("ASSEMBLY COMPLETED! Target Reached.")
            self.flag_target_reached = True

def main(args=None):
    rclpy.init(args=args)
    node = FlexivAssemblyNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectory
import tf2_ros
from flexiv_msgs.msg import RobotStates # Your custom message

DEBUG = False

class SimAdapter(Node):
    def __init__(self):
        super().__init__("flexiv_sim_adapter")
        self.step_counter = 0

        # CONFIG
        self.SN = "" # MATCH YOUR SN
        self.BASE_LINK = "base_link" # Make sure this matches your USD/URDF base name
        self.TCP_LINK = "flange"
        
        # 1. Pubs/Subs
        self.pub_states = self.create_publisher(RobotStates, f"/{self.SN}/flexiv_robot_states", 10) if self.SN else self.create_publisher(RobotStates, "/flexiv_robot_states", 10)
        self.pub_sim_cmd = self.create_publisher(JointState, "/joint_command", 10)
        
        self.sub_wrench = self.create_subscription(WrenchStamped, "/sn/external_wrench_in_world", self.wrench_cb, 10)
        self.sub_joints = self.create_subscription(JointState, "/joint_states", self.joint_cb, 10)
        self.sub_policy_cmd = self.create_subscription(JointTrajectory, "/rizon_arm_controller/joint_trajectory", self.cmd_cb, 10)
        
        # 2. TF Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_wrench = WrenchStamped()
        self.latest_joints = JointState()
        
        self.create_timer(0.02, self.publish_super_message)
        print("[ADAPTER] Running. Bridging Sim <-> Policy.")

    def wrench_cb(self, msg): 
        self.latest_wrench = msg
        if DEBUG and self.step_counter % 5e12 == 0:
            self.get_logger().info(f'Received Wrench Message: {msg}')

    def joint_cb(self, msg): 
        self.latest_joints = msg
        if DEBUG and self.step_counter % 5e12 == 0:
            self.get_logger().info(f'Received Joint State Message: {msg}')

    def cmd_cb(self, msg):
        # Policy (Trajectory) -> Sim (JointState)
        if not msg.points: return
        sim_msg = JointState()
        sim_msg.name = msg.joint_names 
        sim_msg.position = msg.points[0].positions 
        self.pub_sim_cmd.publish(sim_msg)
        self.step_counter += 1

    def publish_super_message(self):
        msg = RobotStates()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world" # Real Flexiv calls its base "world"

        # 1. Joints
        if self.latest_joints.position:
            pos = list(self.latest_joints.position)
            vel = list(self.latest_joints.velocity) if self.latest_joints.velocity else [0.0]*7
            # Map joints if necessary, or pass through
            msg.q = [float(x) for x in pos[:7]]
            msg.dq = [float(x) for x in vel[:7]]

        # 2. Wrench (Already World Frame from Sim)
        msg.ext_wrench_in_world = self.latest_wrench

        # 3. TCP Pose [CRITICAL FIX]
        # We lookup BASE_LINK -> FLANGE, not WORLD -> FLANGE.
        # This removes the effect of where the robot is placed in the sim room.
        try:
            t = self.tf_buffer.lookup_transform(self.BASE_LINK, self.TCP_LINK, rclpy.time.Time())
            
            msg.tcp_pose.pose.position.x = t.transform.translation.x
            msg.tcp_pose.pose.position.y = t.transform.translation.y
            msg.tcp_pose.pose.position.z = t.transform.translation.z
            msg.tcp_pose.pose.orientation = t.transform.rotation
        except Exception as e:
            # Uncomment to debug if frames are missing
            print(f"TF Lookup failed: {e}") 
            pass 

        self.pub_states.publish(msg)

def main():
    rclpy.init()
    node = SimAdapter()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
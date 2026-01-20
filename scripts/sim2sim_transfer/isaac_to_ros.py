# FILE: isaac_to_ros.py
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState # <--- AGGIUNTO
from geometry_msgs.msg import WrenchStamped
import socket
import json

try:
    from flexiv_msgs.msg import RobotStates
    FLEXIV_AVAILABLE = True
except ImportError:
    FLEXIV_AVAILABLE = False
    print("[WARN] flexiv_msgs non trovato. Pubblicherò solo topic standard.")

# CONFIGURAZIONE RETE
LISTEN_IP = "0.0.0.0" 
TARGET_IP = "127.0.0.1" 
UDP_PORT_RECV = 50001 
UDP_PORT_SEND = 50002 

DEBUG = True

# NOMI GIUNTI (Standard Flexiv)
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

class Sim2SimBridge(Node):
    def __init__(self):
        super().__init__('sim2sim_bridge')
        
        # 1. CUSTOM: Quello che usa la tua policy
        if FLEXIV_AVAILABLE:
            self.pub_states = self.create_publisher(RobotStates, "/flexiv_robot_states", 10)
        
        # 2. STANDARD: Quello che serve per il calcolo Jacobiano/TF
        self.pub_joints = self.create_publisher(JointState, "/joint_states", 10)
        
        # 3. WRENCH: Come sul reale
        self.pub_wrench = self.create_publisher(WrenchStamped, "/sn/external_wrench_in_world", 10)
        
        # SUBSCRIBER
        self.sub_traj = self.create_subscription(
            JointTrajectory, "/rizon_arm_controller/joint_trajectory", self.traj_callback, 10
        )
        
        # UDP SETUP
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock_recv.bind((LISTEN_IP, UDP_PORT_RECV))
            self.get_logger().info(f"✅ Bridge Ready. Pubblico RobotStates + JointStates.")
        except Exception as e:
            self.get_logger().error(f"❌ Bind Error: {e}")
            raise e

        self.sock_recv.setblocking(False)
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.create_timer(0.01, self.bridge_loop)

    def traj_callback(self, msg):
        if not msg.points: return
        try:
            positions = msg.points[0].positions
            if DEBUG:
                self.get_logger().info(f"✅ Ricevuto Trajectory: {positions}")
            data = json.dumps(list(positions)[:7]).encode('utf-8')
            self.sock_send.sendto(data, (TARGET_IP, UDP_PORT_SEND))
        except Exception as e:
            if DEBUG:
                self.get_logger().error(f"❌ Trajectory Error: {e}")

    def bridge_loop(self):
        latest_data = None
        try:
            while True:
                data, _ = self.sock_recv.recvfrom(16384)
                latest_data = data
        except BlockingIOError:
            pass
        
        if latest_data:
            try:
                d = json.loads(latest_data.decode('utf-8'))
                now = self.get_clock().now().to_msg()
                
                # --- A. Pubblica /joint_states (NUOVO) ---
                # Questo sblocca il calcolo dello Jacobiano nel nodo assembly
                js = JointState()
                js.header.stamp = now
                js.name = JOINT_NAMES
                js.position = d['joint_pos']
                js.velocity = d['joint_vel']
                self.pub_joints.publish(js)

                # --- B. Pubblica Wrench ---
                wr = WrenchStamped()
                wr.header.stamp = now
                wr.header.frame_id = "world"
                wr.wrench.force.x = d['wrench'][0]
                wr.wrench.force.y = d['wrench'][1]
                wr.wrench.force.z = d['wrench'][2]
                wr.wrench.torque.x = d['wrench'][3]
                wr.wrench.torque.y = d['wrench'][4]
                wr.wrench.torque.z = d['wrench'][5]
                self.pub_wrench.publish(wr)

                # --- C. Pubblica RobotStates (CUSTOM) ---
                if FLEXIV_AVAILABLE:
                    msg = RobotStates()
                    msg.header.stamp = now
                    msg.header.frame_id = "world"
                    
                    msg.tcp_pose.pose.position.x = d['tcp_pos'][0]
                    msg.tcp_pose.pose.position.y = d['tcp_pos'][1]
                    msg.tcp_pose.pose.position.z = d['tcp_pos'][2]
                    msg.tcp_pose.pose.orientation.w = d['tcp_quat'][0]
                    msg.tcp_pose.pose.orientation.x = d['tcp_quat'][1]
                    msg.tcp_pose.pose.orientation.y = d['tcp_quat'][2]
                    msg.tcp_pose.pose.orientation.z = d['tcp_quat'][3]
                    
                    msg.ext_wrench_in_world = wr
                    msg.q = d['joint_pos']
                    msg.dq = d['joint_vel']
                    
                    self.pub_states.publish(msg)
                
            except Exception as e:
                pass

def main():
    rclpy.init()
    node = Sim2SimBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
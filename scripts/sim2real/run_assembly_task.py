# FILE: scripts/sim2real/run_assembly_task.py
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import os
import pinocchio as pin

# Messaggi ROS
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# --- IMPORT CRITICO FLEXIV ---
try:
    from flexiv_msgs.msg import RobotStates
    FLEXIV_IMPORTED = True
except ImportError:
    FLEXIV_IMPORTED = False
    print("\n\n🔴 [ERRORE FATALE] flexiv_msgs non trovato!")
    print("Non posso ricevere Pose e Wrench. Fai 'source install/setup.bash'.\n\n")

# Policy
from robots.rizon.assembly import FlexivGearAssemblyPolicy

# --- CONFIGURAZIONE ---
URDF_PATH = "robots/rizon4s_kinematics.urdf"
CONTROL_FREQ = 15.0

class FlexivAssemblyNode(Node):
    def __init__(self):
        super().__init__("flexiv_assembly_node")
        
        self.dt = 1.0 / CONTROL_FREQ
        self.joint_names_ordered = [f"joint{i}" for i in range(1, 8)]

        # Variabili Stato
        self.robot_state = None      # Pose/Wrench
        self.current_q = None        # Giunti
        self.policy = FlexivGearAssemblyPolicy()

        # --- SETUP PINOCCHIO ---
        if not os.path.exists(URDF_PATH):
            self.get_logger().error(f"URDF MANCANTE: {os.path.abspath(URDF_PATH)}")
            raise FileNotFoundError("Manca il file rizon4s_kinematics.urdf")
            
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId("flange") if self.model.existFrame("flange") else self.model.nframes - 1
        
        # --- DEFINIZIONE QoS (BEST EFFORT) ---
        # Questo è il trucco per leggere qualsiasi topic (Sim o Real)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- SUBSCRIBERS ---
        # 1. Joint States
        self.sub_joints = self.create_subscription(
            JointState, 
            "/joint_states", 
            self.cb_joints, 
            qos_profile  # <--- QoS PERMISSIVO
        )
        
        # 2. Robot States
        if FLEXIV_IMPORTED:
            self.sub_states = self.create_subscription(
                RobotStates, 
                "/flexiv_robot_states", 
                self.cb_states, 
                qos_profile # <--- QoS PERMISSIVO
            )
        
        # Publisher
        self.pub_traj = self.create_publisher(JointTrajectory, "/rizon_arm_controller/joint_trajectory", 1)

        self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("✅ Nodo Avviato. In attesa dei dati...")

    def cb_states(self, msg):
        if self.robot_state is None:
            self.get_logger().info("--> Ricevuto primo RobotStates!")
        self.robot_state = msg

    def cb_joints(self, msg):
        if self.current_q is None:
            self.get_logger().info("--> Ricevuto primo JointState!")
        try:
            q_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            q_ordered = [q_map[name] for name in self.joint_names_ordered]
            self.current_q = np.array(q_ordered)
        except KeyError:
            pass

    def control_loop(self):
        # --- DIAGNOSTICA (Ti dice cosa manca) ---
        missing = []
        if self.robot_state is None: missing.append("/flexiv_robot_states")
        if self.current_q is None: missing.append("/joint_states")
        
        if missing:
            # Stampa ogni 2 secondi per non spammare
            self.get_logger().warn(f"Sto aspettando: {missing}", throttle_duration_sec=2.0)
            return

        # --- 1. JACOBIANO ---
        pin.forwardKinematics(self.model, self.data, self.current_q)
        pin.computeJointJacobians(self.model, self.data, self.current_q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(self.model, self.data, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        # --- 2. INPUT POLICY ---
        pos = np.array([
            self.robot_state.tcp_pose.pose.position.x,
            self.robot_state.tcp_pose.pose.position.y,
            self.robot_state.tcp_pose.pose.position.z
        ])
        quat = np.array([
            self.robot_state.tcp_pose.pose.orientation.x,
            self.robot_state.tcp_pose.pose.orientation.y,
            self.robot_state.tcp_pose.pose.orientation.z,
            self.robot_state.tcp_pose.pose.orientation.w
        ])
        wrench = np.array([
            self.robot_state.ext_wrench_in_world.wrench.force.x,
            self.robot_state.ext_wrench_in_world.wrench.force.y,
            self.robot_state.ext_wrench_in_world.wrench.force.z
        ])

        # --- 3. INFERENZA ---
        raw_action = self.policy.compute_action(pos, quat, wrench)

        # --- 4. DIFF-IK ---
        pos_scale = self.policy.pos_action_bounds 
        rot_scale = self.policy.rot_action_bounds 
        v_lin = (raw_action[0:3] * pos_scale) / self.dt
        v_ang = (raw_action[3:6] * rot_scale) / self.dt
        target_twist = np.concatenate([v_lin, v_ang])

        dls_lambda = 0.05
        J_T = J.T
        J_pinv = J_T @ np.linalg.inv(J @ J_T + dls_lambda**2 * np.eye(6))
        q_dot = J_pinv @ target_twist
        q_cmd = self.current_q + q_dot * self.dt

        # --- 5. INVIO ---
        self.publish_cmd(q_cmd)

    def publish_cmd(self, q_target):
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.joint_names_ordered
        pt = JointTrajectoryPoint()
        pt.positions = q_target.tolist()
        pt.time_from_start = Duration(seconds=self.dt).to_msg()
        traj.points.append(pt)
        self.pub_traj.publish(traj)

def main():
    rclpy.init()
    node = FlexivAssemblyNode()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
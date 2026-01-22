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
from geometry_msgs.msg import WrenchStamped 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Import Flexiv
try:
    from flexiv_msgs.msg import RobotStates
    FLEXIV_IMPORTED = True
except ImportError:
    FLEXIV_IMPORTED = False
    print("\n⚠️ [WARNING] flexiv_msgs non trovato.\n")

from robots.rizon.assembly import FlexivGearAssemblyPolicy

# --- CONFIGURAZIONE ---
URDF_PATH = "robots/rizon4s_kinematics.urdf" 
CONTROL_FREQ = 15.0 
DEBUG = True
SUCCESS_THRESHOLD = 0.99

class FlexivAssemblyNode(Node):
    def __init__(self):
        super().__init__("flexiv_assembly_node")

        self.task_completed = False
        
        self.dt = 1.0 / CONTROL_FREQ
        self.joint_names_ordered = [f"joint{i}" for i in range(1, 8)]

        # Variabili Stato
        self.robot_state = None      
        self.current_q = None        
        self.policy = FlexivGearAssemblyPolicy()

        # TARE Variables
        self.tare_steps = 20
        self.tare_counter = 0
        self.wrench_bias = np.zeros(3)
        self.is_tared = False

        self.step_count = 0

        # --- SETUP PINOCCHIO ---
        if not os.path.exists(URDF_PATH):
            self.get_logger().error(f"URDF MANCANTE: {os.path.abspath(URDF_PATH)}")
            raise FileNotFoundError("Manca il file rizon4s_kinematics.urdf")
            
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId("flange")
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_joints = self.create_subscription(JointState, "/joint_states", self.cb_joints, qos_profile)
        
        if FLEXIV_IMPORTED:
            self.sub_states = self.create_subscription(RobotStates, "/flexiv_robot_states", self.cb_states, qos_profile)
        
        # Sottoscrizione Wrench diretto (Backup per Sim)
        self.sub_wrench = self.create_subscription(WrenchStamped, "/sn/external_wrench_in_world", self.cb_wrench_direct, qos_profile)
        self.last_wrench_msg = None

        self.pub_traj = self.create_publisher(JointTrajectory, "/rizon_arm_controller/joint_trajectory", 1)

        self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(f"✅ Nodo Avviato. In attesa di TARE ({self.tare_steps} steps)...")

    def cb_states(self, msg):
        self.robot_state = msg

    def cb_wrench_direct(self, msg):
        self.last_wrench_msg = msg

    def cb_joints(self, msg):
        try:
            q_map = {name: pos for name, pos in zip(msg.name, msg.position)}
            q_ordered = [q_map[name] for name in self.joint_names_ordered]
            self.current_q = np.array(q_ordered)
        except KeyError:
            pass

    def control_loop(self):

        if self.task_completed:
            return

        # 1. Check Data Availability
        pose_available = False
        wrench_available = False
        
        curr_pos = np.zeros(3)
        curr_quat = np.array([1,0,0,0]) # wxyz
        curr_wrench = np.zeros(3)

        if self.robot_state is not None:
            pose_available = True
            wrench_available = True
            curr_pos = np.array([
                self.robot_state.tcp_pose.pose.position.x,
                self.robot_state.tcp_pose.pose.position.y,
                self.robot_state.tcp_pose.pose.position.z
            ])
            q_msg = self.robot_state.tcp_pose.pose.orientation
            curr_quat = np.array([q_msg.w, q_msg.x, q_msg.y, q_msg.z])
            
            curr_wrench = np.array([
                self.robot_state.ext_wrench_in_world.wrench.force.x,
                self.robot_state.ext_wrench_in_world.wrench.force.y,
                self.robot_state.ext_wrench_in_world.wrench.force.z
            ])
        
        # Fallback se usi lo script Isaac che pubblica WrenchStamped separatamente
        if not wrench_available and self.last_wrench_msg is not None:
            wrench_available = True
            curr_wrench = np.array([
                self.last_wrench_msg.wrench.force.x,
                self.last_wrench_msg.wrench.force.y,
                self.last_wrench_msg.wrench.force.z
            ])

        if self.current_q is None or not pose_available:
            if self.step_count % 30 == 0: self.get_logger().warn("Waiting for Pose/Joints...")
            return

        # --- 2. TARE PROCEDURE (AZZERAMENTO) ---
        if not self.is_tared:
            self.wrench_bias += curr_wrench
            self.tare_counter += 1
            if self.tare_counter >= self.tare_steps:
                self.wrench_bias /= self.tare_steps
                self.is_tared = True
                print(f"\n⚖️  SENSOR TARED! Bias detected: {np.array2string(self.wrench_bias, precision=2)}")
                print("➡️  Starting Policy Control now.\n")
            return 

        # Applica il Tare
        wrench_cleaned = curr_wrench - self.wrench_bias

        # --- 3. POLICY INFERENCE ---
        target_twist, success_score = self.policy.compute_twist(curr_pos, curr_quat, wrench_cleaned)

        # --- 4. DEBUG (CON EMOJI) ---
        if DEBUG and (self.step_count % 100 == 0 or self.step_count < 3):
            dist = np.linalg.norm(curr_pos - self.policy.fixed_pos)
            
            # Formattazione stringhe per allineamento
            p_str = np.array2string(curr_pos, precision=4, suppress_small=True)
            w_str = np.array2string(wrench_cleaned, precision=2, suppress_small=True)
            t_str = np.array2string(target_twist[:3], precision=4, suppress_small=True)
            
            print(f"\n--- 🐛 DEBUG STEP {self.step_count} ---")
            print(f"📍 TCP Current:  {p_str}")
            print(f"📏 Dist to Goal: {dist:.4f} m")
            print(f"🚀 Twist Linear: {t_str}")
            print(f"💪 Force Clean:  {w_str} N")
            print("-----------------------------")

        # 4. CHECK SUCCESS (STOP CONDITION)
        if success_score > SUCCESS_THRESHOLD:
            print(f"\n🎉 SUCCESS DETECTED! Score: {success_score:.4f} > {SUCCESS_THRESHOLD}")
            print(f"🛑 Stopping Robot Commands at Step: {self.step_count}")
            self.task_completed = True
            # Opzionale: Mandare un ultimo comando con velocità zero o la posizione corrente per "freezare"
            self.publish_cmd(self.current_q) 
            self.policy.compute_twist(curr_pos, curr_quat, wrench_cleaned, self.task_completed)
            return

        self.step_count += 1

        # --- 5. DIFF IK ---
        pin.forwardKinematics(self.model, self.data, self.current_q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, self.current_q)
        J = pin.getFrameJacobian(self.model, self.data, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        dls_lambda = 0.05
        J_T = J.T
        J_pinv = J_T @ np.linalg.inv(J @ J_T + dls_lambda**2 * np.eye(6))
        
        q_dot = J_pinv @ target_twist
        q_cmd = self.current_q + q_dot * self.dt
        
        max_q_step = 0.0075
        q_cmd = np.clip(q_cmd, self.current_q - max_q_step, self.current_q + max_q_step)

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
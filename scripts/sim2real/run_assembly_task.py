# FILE: scripts/sim2real/run_assembly_task.py
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import os
import pinocchio as pin
import matplotlib.pyplot as plt  # Aggiunto per i plot

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
CONTROL_FREQ = 60.0 
DEBUG = True
SUCCESS_THRESHOLD = 0.9995
#SUCCESS_THRESHOLD = 0.98
serial_number= 'Rizon4s-063126'

class FlexivAssemblyNode(Node):
    def __init__(self):
        super().__init__("flexiv_assembly_node")

        self.task_completed = False

        self.dt = 1.0 / CONTROL_FREQ
        if serial_number is None:
           self.joint_names_ordered = [f"joint{i}" for i in range(1, 8)]
        else:
            self.joint_names_ordered = [f"{serial_number}_joint{i}" for i in range(1, 8)]

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

        # --- DATA LOGGING ---
        # Accumulatori per i plot finali
        self.log_steps = []
        self.log_actions = []       # Twist [v_lin, v_ang]
        self.log_forces = []        # Wrench [fx, fy, fz]
        self.log_scores = []        # Success probability

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

        #self.sub_joints = self.create_subscription(JointState, "/joint_states", self.cb_joints, qos_profile)
        self.sub_joints = self.create_subscription(JointState, "flexiv_arm/joint_states", self.cb_joints, qos_profile)

        if FLEXIV_IMPORTED and serial_number is None:
                self.sub_states = self.create_subscription(RobotStates, "/flexiv_robot_states", self.cb_states, qos_profile)
        elif FLEXIV_IMPORTED and serial_number:
                self.sub_states = self.create_subscription(RobotStates, f"{serial_number.replace('-','_')}/flexiv_robot_states", self.cb_states, qos_profile)

        # Sottoscrizione Wrench diretto (Backup per Sim)
        if serial_number is None:
            self.sub_wrench = self.create_subscription(WrenchStamped, "/sn/external_wrench_in_world", self.cb_wrench_direct, qos_profile)
        else:
            self.sub_wrench = self.create_subscription(WrenchStamped, f"{serial_number.replace('-','_')}/external_wrench_in_world", self.cb_wrench_direct, qos_profile)
        self.last_wrench_msg = None

        self.pub_traj = self.create_publisher(JointTrajectory, "/rizon_arm_controller/joint_trajectory", 1)

        self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(f"✅ Nodo Avviato. In attesa di TARE ({self.tare_steps} steps)...")

    def cb_states(self, msg):
        # self.get_logger().info(f"✅ Received message({msg}")

        self.robot_state = msg

    def cb_wrench_direct(self, msg):
        # self.get_logger().info(f"✅ Received wrench")

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
            # self.get_logger().info(f"✅ Robot state is available")
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
            # self.get_logger().info(f"✅ Qui ci sono pure")
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

        # --- LOGGING PER PLOT ---
        self.log_steps.append(self.step_count)
        self.log_actions.append(target_twist)      # [vx, vy, vz, wx, wy, wz]
        self.log_forces.append(wrench_cleaned)     # [fx, fy, fz]
        self.log_scores.append(success_score)

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
            # Chiusura nodo gestita nel main per permettere il plot
            raise SystemExit 

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

    def plot_results(self):
        """Genera i plot richiesti a fine esecuzione"""
        if not self.log_steps:
            print("Nessun dato registrato da plottare.")
            return

        print("\n📊 Generazione grafici in corso...")

        steps = np.array(self.log_steps)
        actions = np.array(self.log_actions) # Shape (N, 6)
        forces = np.array(self.log_forces)   # Shape (N, 3)
        scores = np.array(self.log_scores)   # Shape (N,)

        # Creazione figura con 3 subplot
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # 1. Andamento Azioni (Twist)
        # Plot Linear Velocity
        axs[0].plot(steps, actions[:, 0], label='Vx', linestyle='-', alpha=0.8)
        axs[0].plot(steps, actions[:, 1], label='Vy', linestyle='-', alpha=0.8)
        axs[0].plot(steps, actions[:, 2], label='Vz', linestyle='-', alpha=0.8)
        # Plot Angular Velocity (tratteggiato per distinguere)
        axs[0].plot(steps, actions[:, 3], label='Wx', linestyle='--', alpha=0.5)
        axs[0].plot(steps, actions[:, 4], label='Wy', linestyle='--', alpha=0.5)
        axs[0].plot(steps, actions[:, 5], label='Wz', linestyle='--', alpha=0.5)
        axs[0].set_ylabel("Action (Twist m/s & rad/s)")
        axs[0].set_title("1. Andamento delle Azioni (Twist) nel tempo")
        axs[0].legend(loc='upper right', ncol=2)
        axs[0].grid(True, alpha=0.3)

        # 2. Andamento Forze
        axs[1].plot(steps, forces[:, 0], label='Fx', color='r', alpha=0.7)
        axs[1].plot(steps, forces[:, 1], label='Fy', color='g', alpha=0.7)
        axs[1].plot(steps, forces[:, 2], label='Fz', color='b', alpha=0.7)
        axs[1].set_ylabel("Force (N)")
        axs[1].set_title("2. Andamento delle Forze Misurate (World Frame)")
        axs[1].legend(loc='upper right')
        axs[1].grid(True, alpha=0.3)

        # 3. Iterazioni e Successo
        axs[2].plot(steps, scores, label='Success Score', color='purple', linewidth=2)
        axs[2].axhline(y=SUCCESS_THRESHOLD, color='k', linestyle='--', label='Threshold')
        axs[2].set_ylabel("Probability")
        axs[2].set_xlabel("Steps (Iterations)")
        axs[2].set_title(f"3. Success Score Evolution (Total Steps: {steps[-1]})")
        axs[2].legend(loc='lower right')
        axs[2].grid(True, alpha=0.3)
        axs[2].set_ylim([-0.1, 1.1])

        plt.tight_layout()
        plt.show()

def main():
    rclpy.init()
    node = FlexivAssemblyNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Interruzione rilevata. Chiusura nodo...")
    finally:
        # Esegue il plot sia in caso di successo (SystemExit) che di Ctrl+C
        node.plot_results()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
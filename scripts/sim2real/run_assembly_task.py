# FILE: scripts/sim2real/run_assembly_task.py
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import os
import csv
import pinocchio as pin
import matplotlib.pyplot as plt # [NEW] Plotting

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
    print("\n⚠️ [WARNING] flexiv_msgs not found.\n")

from robots.rizon.assembly import FlexivGearAssemblyPolicy

# --- CONFIGURAZIONE ---
DEBUG = True
SIMULATED = False
FORCE_SCALE = (12.5/125) # used only in simulation to scale forces
seed=0 # one of the 3 seeds determining the initial state


URDF_PATH = "robots/rizon4s_kinematics.urdf" 
ROOT_LOG_FOLDER = "logs"
CSV_FILENAME =  ROOT_LOG_FOLDER + "/sim2real_results.csv"
PLOTS_FOLDER = ROOT_LOG_FOLDER + "/plots"

if SIMULATED:
    serial_number = None
else:
    serial_number = "Rizon4s-063126"

# control parameters
CONTROL_FREQ = 60.0
SUCCESS_THRESHOLD = 0.99 # threshold for considering the task successful and stopping the control



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
        self.policy = FlexivGearAssemblyPolicy(seed)

        # TARE Variables
        self.tare_steps = 20
        self.tare_counter = 0
        self.wrench_bias = np.zeros(3)
        self.is_tared = False

        self.step_count = 0

        # [NEW] Data Logging
        self.log_steps = []
        self.log_actions = []
        self.log_forces = []
        self.log_scores = []
        self.tcp_offset_z = 0.19909 # [m] Offset TCP vs Flange

        # [NEW] Metrics for CSV
        self.metric_path_length = 0.0
        self.metric_cumulative_force = 0.0
        self.metric_max_force = 0.0
        self.last_pos_for_metric = None
        self.data_saved = False

        # [NEW] Episode ID & File Setup
        if not os.path.exists(PLOTS_FOLDER):
            os.makedirs(PLOTS_FOLDER)

        self.episode_id = 1
        if os.path.exists(CSV_FILENAME):
            with open(CSV_FILENAME, 'r') as f:
                # Count lines to determine ID (header is line 1)
                self.episode_id = sum(1 for _ in f)

        # --- SETUP PINOCCHIO ---
        if not os.path.exists(URDF_PATH):
            self.get_logger().error(f"Missing URDF: {os.path.abspath(URDF_PATH)}")
            raise FileNotFoundError("Missing file rizon4s_kinematics.urdf")

        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId("flange")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        if serial_number is None:
            self.sub_joints = self.create_subscription(JointState, "/joint_states", self.cb_joints, qos_profile)

        else:
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
        self.get_logger().info(f"✅ Node started. Episode ID: {self.episode_id}. Waiting for TARE calibration...")

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
            # self.get_logger().info(f"✅ Robot state is available")
            pose_available = True
            wrench_available = True
            curr_pos = np.array([
                self.robot_state.flange_pose.pose.position.x,
                self.robot_state.flange_pose.pose.position.y,
                self.robot_state.flange_pose.pose.position.z
            ])
            q_msg = self.robot_state.tcp_pose.pose.orientation
            curr_quat = np.array([q_msg.w, q_msg.x, q_msg.y, q_msg.z])

            curr_wrench = np.array([
                self.robot_state.ext_wrench_in_world.wrench.force.x,
                self.robot_state.ext_wrench_in_world.wrench.force.y,
                self.robot_state.ext_wrench_in_world.wrench.force.z
            ])

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
        # Flange to TCP
        M_world_flange = None
        rot_mat = pin.Quaternion(curr_quat[0], curr_quat[1], curr_quat[2], curr_quat[3]).toRotationMatrix()
        M_world_flange = pin.SE3(rot_mat, curr_pos)
        M_flange_tcp = pin.SE3(np.eye(3), np.array([0.0, 0.0, self.tcp_offset_z]))
        M_world_tcp = M_world_flange * M_flange_tcp
        curr_pos = M_world_tcp.translation
        quat_pin = pin.Quaternion(M_world_tcp.rotation)
        curr_quat = np.array([quat_pin.w, quat_pin.x, quat_pin.y, quat_pin.z])

        # --- 2. TARE PROCEDURE ---
        if not self.is_tared:
            self.wrench_bias += curr_wrench
            self.tare_counter += 1
            if self.tare_counter >= self.tare_steps:
                self.wrench_bias /= self.tare_steps
                self.is_tared = True
                print(f"\n⚖️  SENSOR TARED! Bias detected: {np.array2string(self.wrench_bias, precision=2)}")
                print("➡️  Starting Policy Control now.\n")
            return 

        wrench_cleaned = curr_wrench - self.wrench_bias
        if SIMULATED:
            wrench_cleaned *= FORCE_SCALE

        # Metrics Update (Real-time)
        force_norm = np.linalg.norm(wrench_cleaned)
        if force_norm > self.metric_max_force:
            self.metric_max_force = force_norm

        self.metric_cumulative_force += force_norm * self.dt

        if self.last_pos_for_metric is not None:
            self.metric_path_length += np.linalg.norm(curr_pos - self.last_pos_for_metric)
        self.last_pos_for_metric = curr_pos

        # --- 3. POLICY INFERENCE ---
        target_twist, success_score = self.policy.compute_twist(curr_pos, curr_quat, wrench_cleaned)

        # Logging
        self.log_steps.append(self.step_count)
        self.log_actions.append(target_twist)
        self.log_forces.append(wrench_cleaned)
        self.log_scores.append(success_score)

        # --- 4. DEBUG  ---
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

            self.publish_cmd(self.current_q) 
            self.policy.compute_twist(curr_pos, curr_quat, wrench_cleaned, self.task_completed)

            # Save CSV Data on Success
            self.save_episode_data(termination_reason="Success", success_flag=True)

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

        max_q_step = 0.0035
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

    def save_episode_data(self, termination_reason, success_flag):
        if self.data_saved: return

        completion_time = self.step_count * self.dt

        # CSV Headers: ID, Seed, Simulated, Success, Termination_Reason, Completion_Time, Path_Length, Max_Force, Cumulative_Force
        file_exists = os.path.exists(CSV_FILENAME)

        with open(CSV_FILENAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["ID", "Seed", "Simulated", "Success", "Termination_Reason", 
                                 "Completion_Time(s)", "Path_Length(m)", "Max_Force_Exerted(N)", "Cumulative_Contact_Force(N)", "Success_threshold", "False_positive"])

            writer.writerow([
                self.episode_id,
                seed,
                SIMULATED,
                success_flag,
                termination_reason,
                f"{completion_time:.4f}",
                f"{self.metric_path_length:.4f}",
                f"{self.metric_max_force:.4f}",
                f"{self.metric_cumulative_force:.4f}",
                f"{SUCCESS_THRESHOLD:.4f}"
            ])

        print(f"\n💾 Episode Data saved to {CSV_FILENAME}")
        self.data_saved = True
        self.plot_results()

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

        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot Linear Velocity
        axs[0].plot(steps, actions[:, 0], label='Vx', linestyle='-', alpha=0.8)
        axs[0].plot(steps, actions[:, 1], label='Vy', linestyle='-', alpha=0.8)
        axs[0].plot(steps, actions[:, 2], label='Vz', linestyle='-', alpha=0.8)
        # Plot Angular Velocity 
        axs[0].plot(steps, actions[:, 3], label='Wx', linestyle='--', alpha=0.5)
        axs[0].plot(steps, actions[:, 4], label='Wy', linestyle='--', alpha=0.5)
        axs[0].plot(steps, actions[:, 5], label='Wz', linestyle='--', alpha=0.5)
        axs[0].set_ylabel("Action (Twist m/s & rad/s)")
        axs[0].set_title(f"Episode {self.episode_id} - Actions")
        axs[0].legend(loc='upper right', ncol=2)
        axs[0].grid(True, alpha=0.3)

        # 2. Forces
        axs[1].plot(steps, forces[:, 0], label='Fx', color='r', alpha=0.7)
        axs[1].plot(steps, forces[:, 1], label='Fy', color='g', alpha=0.7)
        axs[1].plot(steps, forces[:, 2], label='Fz', color='b', alpha=0.7)
        axs[1].set_ylabel("Force (N)")
        axs[1].set_title("2. Forces (World Frame)")
        axs[1].legend(loc='upper right')
        axs[1].grid(True, alpha=0.3)

        # 3. Success Score
        axs[2].plot(steps, scores, label='Success Score', color='purple', linewidth=2)
        axs[2].axhline(y=SUCCESS_THRESHOLD, color='k', linestyle='--', label='Threshold')
        axs[2].set_ylabel("Probability")
        axs[2].set_xlabel("Steps (Iterations)")
        axs[2].set_title(f"3. Success Score Evolution (Total Steps: {steps[-1]})")
        axs[2].legend(loc='lower right')
        axs[2].grid(True, alpha=0.3)
        axs[2].set_ylim([-0.1, 1.1])

        plt.tight_layout()
        plot_path = os.path.join(PLOTS_FOLDER, f"episode_{self.episode_id}.png")
        plt.savefig(plot_path)
        print(f"🖼️  Plot saved to: {plot_path}")
        # plt.show() # Commented out to allow automated running

def main():
    rclpy.init()
    node = FlexivAssemblyNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Interruzione rilevata.")
        # Save aborted data if not already saved
        if not node.task_completed:
             node.save_episode_data(termination_reason="Aborted", success_flag=False)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
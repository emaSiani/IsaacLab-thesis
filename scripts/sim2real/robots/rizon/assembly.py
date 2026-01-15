# FILE: scripts/sim2real/robots/rizon/assembly.py
import numpy as np
import torch

# robot pos deg: [-7,44, -32,7, 24,17, 102,3, -18,16, 40,4, 0.0]   

DEBUG = True


class FlexivGearAssemblyPolicy:
    """
    Policy for Rizon 4s (FORGE Environment).
    Action Space: 7 DOF (Arm Only).
    Gripper: Assumed fixed/grasping.
    """

    def __init__(self):
        # --- PATH CONFIG ---
        self.policy_path = r"robots/rizon/policies/rizon4s_200ep_512envs_halved_orn_noise_policy.pt"
        
        # [CRITICAL] Position of the HOLE in Robot Base Frame
        # You must measure this in the real world!
        #self.fixed_pos = np.array([10.5, 10.0, -80.2]) 
        self.fixed_pos = np.array([0.62804, 0.03906, 0.08477]) 
        self.force_threshold = np.array([1.14]) 

        # --- MODEL LOAD ---
        self.device = torch.device("cpu")
        try:
            self.model = torch.jit.load(self.policy_path, map_location=self.device)
            self.model.eval()
        except Exception as e:
            print(f"[ERROR] Policy load failed: {e}")
            raise e

        # --- CONSTANTS ---
        self.dt = 0.02 
        self.decimation = 2 
        self.action_scale = 0.5 
        self.default_joint_pos = np.array([-0.13, -0.5707, 0.422, 1.7854, -0.317, 0.705, 0.0], dtype=np.float32)

        # --- STATE ---
        self.prev_ee_pos = None
        self.prev_ee_quat = None 
        self.prev_action = np.zeros(7, dtype=np.float32) # FIX: Size 7
        self.step_counter = 0

        # RNN Init
        self.rnn_units = 1024 
        h = torch.zeros((2, 1, self.rnn_units), device=self.device)
        c = torch.zeros((2, 1, self.rnn_units), device=self.device)
        self.rnn_states = (h, c)

    def compute_action(self, current_ee_pos, current_ee_quat, current_force):
        """
        Args:
            current_ee_pos: Gear Tip Position [x,y,z]
            current_ee_quat: Gear Orientation [w,x,y,z]
        """
        # 1. Initialization
        if self.prev_ee_pos is None:
            self.prev_ee_pos = current_ee_pos
            self.prev_ee_quat = current_ee_quat
            return np.zeros(7), 0.0 # Return 0 velocity, 0.0 width (Closed)

        # 2. Decimation
        if self.step_counter % self.decimation != 0:
            self.step_counter += 1
            arm_action = self.default_joint_pos + (self.prev_action * self.action_scale)
            return arm_action, 0.0

        # 3. Observation Construction
        lin_vel = (current_ee_pos - self.prev_ee_pos) / self.dt
        
        # Ang Vel Approx
        if np.dot(current_ee_quat, self.prev_ee_quat) < 0:
            current_ee_quat = -current_ee_quat
        q_diff = current_ee_quat - self.prev_ee_quat
        ang_vel = q_diff[1:] * 2.0 / self.dt 

        pos_rel = current_ee_pos - self.fixed_pos

        # Masking (Indices 3,4 of 7-DOF)
        masked_prev_actions = self.prev_action.copy()
        masked_prev_actions[3:5] = 0.0

        # Tensor Build
        quat_xyzw = np.array([current_ee_quat[1], current_ee_quat[2], current_ee_quat[3], current_ee_quat[0]])

        obs_vec = np.concatenate([
            current_ee_pos,     # 3
            pos_rel,            # 3
            quat_xyzw,          # 4
            lin_vel,            # 3
            ang_vel,            # 3
            current_force,      # 3
            self.force_threshold, # 1
            masked_prev_actions # 7 (Corrected)
        ]).astype(np.float32)

        # if debug print observation vector every 100 steps
        if DEBUG and (self.step_counter % 100 == 0 or self.step_counter == 0):
            print(f"[DEBUG] Step: {self.step_counter} | Observation Vector: {obs_vec}")

        # 4. Inference
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_vec, device=self.device).unsqueeze(0)
            action_t, self.rnn_states = self.model(obs_t, self.rnn_states)
            raw_action = action_t.cpu().numpy().flatten()

        # 5. Update
        self.prev_action = raw_action
        self.prev_ee_pos = current_ee_pos
        self.prev_ee_quat = current_ee_quat
        self.step_counter += 1

        # 6. Output
        arm_target = self.default_joint_pos + (raw_action * self.action_scale)
        gripper_target = 0.0 # Always enforce Closed

        return arm_target, gripper_target
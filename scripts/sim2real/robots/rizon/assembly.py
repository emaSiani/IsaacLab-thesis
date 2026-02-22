# FILE: scripts/sim2real/robots/rizon/assembly.py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

DEBUG = True
SIMULATION = False
ROTATION = False
if SIMULATION:
    FORCE_SCALE = (12.5/150)
else:
    FORCE_SCALE = 1.0

# INITIAL ROBOT POSE
# deg: [-27.46, -45.05, 52.05, 92.91, -39.24, 28.07, -150.73]  


class FlexivGearAssemblyPolicy:
    def __init__(self, seed=0):
        # --- PATH CONFIG ---
        if not ROTATION:
            self.policy_path = r"robots/rizon/policies/policy.pt"
        else:
            self.policy_path = r"robots/rizon/policies/rotation_policy.pt"

        match seed:
            case 0: 
                self.fixed_pos = np.array([0.65091, 0.04118, 0.11824]) 
            case 1:
                self.fixed_pos = np.array([0.64091, 0.04118, 0.11824]) 
            case 2:
                self.fixed_pos = np.array([0.65091, 0.02016, 0.11824]) 
            case 3: 
                self.fixed_pos = np.array([0.65091, 0.04118, 0.12824]) 

        self.target_yaw =  np.pi / 6
        self.force_threshold = np.array([5.25]) 

        # --- MODEL LOAD ---
        self.device = torch.device("cpu")
        print(f"Loading JIT policy from: {self.policy_path}")
        self.model = torch.jit.load(self.policy_path, map_location=self.device)
        self.model.eval()

        # --- PARAMETRI TRAINING ---
        self.dt = 1.0 / 15.0 # 
        # Nota: self.dt qui è il dt di CONTROLLO (decimato). 

        # Bounds & Thresholds
        self.pos_action_bounds = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        self.rot_action_bounds = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Clipping
        self.pos_threshold = 0.02
        self.rot_threshold = 0.097

        # EMA Smoothing 
        self.ema_factor = 0.05

        # FT Smoothing 
        self.ft_smoothing_factor = 0.25

        # --- STATE BUFFERS ---
        self.prev_ee_pos = None
        self.prev_ee_quat = None 
        self.prev_action = np.zeros(7, dtype=np.float32) 
        self.prev_action_smooth = np.zeros(7, dtype=np.float32)

        # Buffer per smoothing forza
        self.force_sensor_world_smooth = np.zeros(3, dtype=np.float32)

        # RNN Init (LSTM)
        self.rnn_units = 1024 
        h = torch.zeros((2, 1, self.rnn_units), device=self.device)
        c = torch.zeros((2, 1, self.rnn_units), device=self.device)
        self.rnn_states = (h, c)

        self.step_counter = 0

    def rotate_force_to_world(self, force_body, quat_xyzw):
        """Ruota la forza dal frame sensore (Body) al frame World"""
        # Se il sensore ti da già forza in World Frame, puoi saltare questo.
        # Ma solitamente i sensori F/T sono montati sulla flangia.
        r = R.from_quat(quat_xyzw)
        force_world = r.apply(force_body)
        return force_world

    def compute_twist(self, current_ee_pos, current_ee_quat, current_force_world_raw, task_completed=False):
        """
        Args:
            current_ee_pos: [x, y, z]
            current_ee_quat: [w, x, y, z] 
            current_force_world_raw: [fx, fy, fz] 
        """
        current_ee_quat[0] = 0.0
        current_ee_quat[3] = 0.0
        current_ee_quat *= -1.0

        if self.prev_ee_pos is None:
            self.prev_ee_pos = current_ee_pos
            self.prev_ee_quat = current_ee_quat
            self.force_sensor_world_smooth = current_force_world_raw 

            pos_rel_start = current_ee_pos - self.fixed_pos

            init_action = np.zeros(7, dtype=np.float32)
            init_action[0:3] = pos_rel_start / self.pos_action_bounds

            init_action[6] = -1.0

            self.prev_action_smooth = init_action
            self.prev_action = init_action
            # ---------------------------------------------------

            return np.zeros(6), 0

        # 2. Compute Input Velocities (Finite Difference)
        lin_vel = (current_ee_pos - self.prev_ee_pos) / self.dt

        #if np.dot(current_ee_quat, self.prev_ee_quat) < 0:
        #    current_ee_quat = -current_ee_quat 

        r_curr = R.from_quat([current_ee_quat[1], current_ee_quat[2], current_ee_quat[3], current_ee_quat[0]])
        r_prev = R.from_quat([self.prev_ee_quat[1], self.prev_ee_quat[2], self.prev_ee_quat[3], self.prev_ee_quat[0]])
        r_diff = r_curr * r_prev.inv()
        rot_vec = r_diff.as_rotvec()
        ang_vel = rot_vec / self.dt

        # Applica Smoothing esponenziale
        alpha = self.ft_smoothing_factor
        self.force_sensor_world_smooth = alpha * current_force_world_raw + (1 - alpha) * self.force_sensor_world_smooth
        current_force_obs = self.force_sensor_world_smooth * FORCE_SCALE
        pos_rel = current_ee_pos - self.fixed_pos
        masked_prev_actions = self.prev_action_smooth.copy()
        masked_prev_actions[3:5] = 0.0 

        # 6. Build Observation
        common_obs = np.concatenate([
            pos_rel,            # 3
            current_ee_quat,    # 4 (w,x,y,z)
            lin_vel,            # 3
            ang_vel,            # 3
            current_force_obs,  # 3 (Smoothed & World Frame)
            self.force_threshold, # 1
        ]).astype(np.float32)

        if ROTATION:
            current_yaw = r_curr.as_euler('zyx')[0] 
            target_yaw_error = self.target_yaw - current_yaw
            target_yaw_error = (target_yaw_error + np.pi) % (2 * np.pi) - np.pi
            common_obs = np.concatenate([
                common_obs, 
                np.array([target_yaw_error], dtype=np.float32)
            ])

        obs_vec = np.concatenate([
            common_obs, 
            masked_prev_actions
        ]).astype(np.float32)


        # check if the force smoothed changes

        if DEBUG and ((self.step_counter % 100 == 0 or self.step_counter < 3) or task_completed): 
            print(f"\n##################### Step: {self.step_counter} OBSERVATION #####################")
            keys = ["pos_rel", "quat", "lin_vel", "ang_vel", "force_smooth", "threshold", "prev_act"]
            vals = [pos_rel, current_ee_quat, lin_vel, ang_vel, current_force_obs, self.force_threshold, masked_prev_actions]

            if ROTATION:
                keys.insert(-1, 'yaw_error')
                vals.insert(-1, np.array([target_yaw_error]))

            for key, val in zip(keys, vals):
                print(f"{key:<15}: {np.array2string(val, precision=4, suppress_small=True)}")
            print("#####################################################################\n")

        if task_completed:
            return

        # 7. Inference
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_vec, device=self.device).unsqueeze(0)
            action_t, self.rnn_states = self.model(obs_t, self.rnn_states)
            action_t = torch.clamp(action_t, min=-1.0, max=1.0)
            raw_action = action_t.cpu().numpy().flatten()

        # 8. Post-Processing (EMA Action)
        self.prev_action = raw_action
        ema = self.ema_factor
        smooth_action = ema * raw_action + (1 - ema) * self.prev_action_smooth
        self.prev_action_smooth = smooth_action


        raw_success_pred = smooth_action[6] 
        # Scaling to [0, 1]
        success_score = (raw_success_pred + 1.0) / 2.0

        if DEBUG and (self.step_counter % 100 == 0 or self.step_counter < 3):
            print(f"🔮 Success Prediction: {raw_success_pred:.2f} -> {success_score:.2f}")

        # 9. FORGE LOGIC: Convert Action to Twist
        pos_action_delta = smooth_action[0:3] * self.pos_action_bounds
        rot_action_delta = smooth_action[3:6] * self.rot_action_bounds
        rot_action_delta[3:5] = 0.0

        target_pos_world = self.fixed_pos + pos_action_delta
        delta_pos = target_pos_world - current_ee_pos
        delta_pos_clipped = np.clip(delta_pos, -self.pos_threshold, self.pos_threshold)

        v_lin_cmd = delta_pos_clipped / self.dt

        delta_rot_clipped = np.clip(rot_action_delta, -self.rot_threshold, self.rot_threshold)
        v_ang_cmd = delta_rot_clipped / self.dt 

        self.prev_ee_pos = current_ee_pos
        self.prev_ee_quat = current_ee_quat

        self.step_counter += 1

        return np.concatenate([v_lin_cmd, v_ang_cmd]), success_score
 
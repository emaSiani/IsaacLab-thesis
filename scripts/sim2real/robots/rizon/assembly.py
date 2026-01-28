# FILE: scripts/sim2real/robots/rizon/assembly.py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

DEBUG = True
SIMULATION = False

# INITIAL ROBOT POSE
# deg: [-7,44, -32,7, 24,17, 102,3, -18,16, 40,4, 0.0]  


# INITIAL ROBOT POSE
# deg: [-7,44, -32,7, 24,17, 102,3, -18,16, 40,4, 0.0]  


class FlexivGearAssemblyPolicy:
    def __init__(self):
        # --- PATH CONFIG ---
        #self.policy_path = r"robots/rizon/policies/rizon4s_200ep_512envs_increase_kp_scale.pt"
        self.policy_path = r"robots/rizon/policies/rizon4s_200ep_512envs_fixed_midpoint_policy.pt"

        # [CRITICAL] Posizione del FIXED ASSET (Bullone) nel frame del robot reale.
<<<<<<< HEAD
        self.fixed_pos = np.array([0.6047, 0.02619, 0.0782]) 
        self.fixed_pos[0] += 0.02025  # offset of the bolt

        self.force_threshold = np.array([1.1]) 
=======
        if SIMULATION: 
            self.fixed_pos = np.array([0.6047, 0.02619, 0.0782]) 
        else:
            self.fixed_pos = np.array([0.62935, 0.03585, 0.0782]) 
            # self.fixed_pos[0] -= 0.0048
            # self.fixed_pos[0] += 0.0176
        #self.fixed_pos[0] += 0.02025  # offset of the bolt

        self.force_threshold = np.array([0.5]) 
>>>>>>> 0abd9770 (28/01/26 - Deployed in REAL)

        # --- MODEL LOAD ---
        self.device = torch.device("cpu")
        print(f"Loading JIT policy from: {self.policy_path}")
        self.model = torch.jit.load(self.policy_path, map_location=self.device)
        self.model.eval()

        # --- PARAMETRI TRAINING ---
        self.dt = 1.0 / 15.0 # ~0.066s (corretto rispetto a 0.02s di simulazione fisica pura)
        # Nota: self.dt qui è il dt di CONTROLLO (decimato). 

        # Bounds & Thresholds
        self.pos_action_bounds = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        self.rot_action_bounds = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Clipping
        self.pos_threshold = 0.02
        self.rot_threshold = 0.097

        # EMA Smoothing (Azioni)
        self.ema_factor = 0.05

        # FT Smoothing (Forza) - Dal config originale ft_smoothing_factor = 0.25
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
            current_ee_quat: [w, x, y, z] (Isaac Order)
            current_force_world_raw: [fx, fy, fz] (GIA' IN WORLD FRAME da ROS)
        """
        # 1. Initialization
        if self.prev_ee_pos is None:
            self.prev_ee_pos = current_ee_pos
            self.prev_ee_quat = current_ee_quat
            # Inizializza il buffer direttamente con il valore raw (senza rotazione)
            self.force_sensor_world_smooth = current_force_world_raw
            return np.zeros(6), 0

        # 2. Compute Input Velocities (Finite Difference)
        lin_vel = (current_ee_pos - self.prev_ee_pos) / self.dt

        if np.dot(current_ee_quat, self.prev_ee_quat) < 0:
            current_ee_quat = -current_ee_quat 

        r_curr = R.from_quat([current_ee_quat[1], current_ee_quat[2], current_ee_quat[3], current_ee_quat[0]])
        r_prev = R.from_quat([self.prev_ee_quat[1], self.prev_ee_quat[2], self.prev_ee_quat[3], self.prev_ee_quat[0]])
        r_diff = r_curr * r_prev.inv()
        rot_vec = r_diff.as_rotvec()
        ang_vel = rot_vec / self.dt

        # 3. FORCE PROCESSING (Solo Smoothing, NIENTE Rotazione)
        # Il topic ROS ext_wrench_in_world è già orientato correttamente.

        # Applica Smoothing esponenziale
        alpha = self.ft_smoothing_factor
        self.force_sensor_world_smooth = alpha * current_force_world_raw + (1 - alpha) * self.force_sensor_world_smooth

        current_force_obs = self.force_sensor_world_smooth

        # 4. Posizione Relativa
        pos_rel = current_ee_pos - self.fixed_pos

        # 5. Prev Actions Masking
        masked_prev_actions = self.prev_action_smooth.copy()
        masked_prev_actions[3:5] = 0.0 

        # 6. Build Observation
        obs_vec = np.concatenate([
            pos_rel,            # 3
            current_ee_quat,    # 4 (w,x,y,z)
            lin_vel,            # 3
            ang_vel,            # 3
            current_force_obs,  # 3 (Smoothed & World Frame)
            self.force_threshold, # 1
            masked_prev_actions # 7
        ]).astype(np.float32)

        # check if the force smoothed changes
        force_changed = not np.allclose(current_force_obs, self.force_sensor_world_smooth)

        if DEBUG and ((self.step_counter % 100 == 0 or self.step_counter < 3) or force_changed or task_completed): 
            print(f"\n##################### Step: {self.step_counter} OBSERVATION #####################")
            keys = ["pos_rel", "quat", "lin_vel", "ang_vel", "force_smooth", "threshold", "prev_act"]
            vals = [pos_rel, current_ee_quat, lin_vel, ang_vel, current_force_obs, self.force_threshold, masked_prev_actions]

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

        # --- ESTRAZIONE SUCCESS PREDICTION ---
        # L'indice 6 è la predizione (-1 fallimento, +1 successo)
        raw_success_pred = smooth_action[6] 
        # Scaliamo da [-1, 1] a [0, 1]
        success_score = (raw_success_pred + 1.0) / 2.0

        if DEBUG and (self.step_counter % 100 == 0 or self.step_counter < 3):
            print(f"🔮 Success Prediction: {raw_success_pred:.2f} -> {success_score:.2f}")

        # 9. FORGE LOGIC: Convert Action to Twist
        pos_action_delta = smooth_action[0:3] * self.pos_action_bounds
        rot_action_delta = smooth_action[3:6] * self.rot_action_bounds

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
 
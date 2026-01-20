# FILE: scripts/sim2real/robots/rizon/assembly.py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

class FlexivGearAssemblyPolicy:
    def __init__(self):
        # --- PATH CONFIG ---
        self.policy_path = r"robots/rizon/policies/rizon4s_200ep_512envs_fixed_midpoint_policy.pt"
        
        # [CRITICAL] Posizione del FIXED ASSET (Bullone) nel frame del robot reale.
        # Devi misurarla con precisione millimetrica!
        self.fixed_pos = np.array([0.61422, 0.03906, 0.06479]) 
        
        # Soglia usata nel training
        self.force_threshold = np.array([5.14]) 

        # --- MODEL LOAD ---
        self.device = torch.device("cpu")
        self.model = torch.jit.load(self.policy_path, map_location=self.device)
        self.model.eval()

        # --- PARAMETRI TRAINING ---
        # Devono essere IDENTICI a `rizon4s_forge_env_cfg.py`
        self.dt = 0.02 
        self.decimation = 8 # Training decimation (120Hz / 8 = 15Hz)
        
        # Scaling actions (da CtrlCfg)
        self.pos_action_bounds = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        self.rot_action_bounds = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # --- STATE ---
        self.prev_ee_pos = None
        self.prev_ee_quat = None 
        # Action buffer size 7 (6 dof + 1 gripper)
        self.prev_action = np.zeros(7, dtype=np.float32) 
        self.step_counter = 0

        # RNN Init (LSTM)
        self.rnn_units = 1024 
        h = torch.zeros((2, 1, self.rnn_units), device=self.device)
        c = torch.zeros((2, 1, self.rnn_units), device=self.device)
        self.rnn_states = (h, c)

    def compute_action(self, current_ee_pos, current_ee_quat, current_force):
        """
        Processa le osservazioni e ritorna l'azione raw normalizzata.
        Args:
            current_ee_pos: [x, y, z] (World Frame)
            current_ee_quat: [w, x, y, z] (Isaac Order)
            current_force: [fx, fy, fz] (World Frame)
        Returns:
            raw_action: np.array(7) - Valori tra -1 e 1
        """
        # 1. Initialization
        if self.prev_ee_pos is None:
            self.prev_ee_pos = current_ee_pos
            self.prev_ee_quat = current_ee_quat
            return np.zeros(7, dtype=np.float32)

        # 2. Decimation Check
        # Nota: Nel real-time loop, chiamiamo questa funzione a ogni ciclo.
        # Se vogliamo rispettare il decimation, aggiorniamo l'azione solo ogni N step.
        # Qui assumiamo che il chiamante gestisca il rate (es. 15Hz) o facciamo update sempre.
        # Per semplicità, eseguiamo sempre l'inferenza assumendo che il nodo ROS giri a ~15-20Hz.

        # 3. Calcolo Velocità (Finite Difference come nel Training)
        lin_vel = (current_ee_pos - self.prev_ee_pos) / self.dt
        
        # Ang Vel Approx
        # Ensure continuity
        if np.dot(current_ee_quat, self.prev_ee_quat) < 0:
            temp_curr_quat = -current_ee_quat
        else:
            temp_curr_quat = current_ee_quat
            
        # Semplice diff per ang vel (sufficiente per policy robuste)
        # Oppure conversione in Axis-Angle diff
        q_diff = temp_curr_quat - self.prev_ee_quat
        ang_vel = q_diff[1:] * 2.0 / self.dt # approx parte vettoriale

        # 4. Posizione Relativa
        pos_rel = current_ee_pos - self.fixed_pos

        # 5. Prev Actions Masking 
        # [CRITICO] Nel training `_get_observations` fa: prev_actions[:, 3:5] = 0.0
        masked_prev_actions = self.prev_action.copy()
        masked_prev_actions[3:5] = 0.0 

        # 6. Costruzione Osservazione
        # Ordine da rizon4s_forge_env_cfg.py:
        # ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel", "ft_force", "force_threshold", "prev_actions"]
        
        obs_vec = np.concatenate([
            pos_rel,            # 3
            current_ee_quat,    # 4 (w,x,y,z)
            lin_vel,            # 3
            ang_vel,            # 3
            current_force,      # 3
            self.force_threshold, # 1
            masked_prev_actions # 7
        ]).astype(np.float32)

        # 7. Inferenza
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_vec, device=self.device).unsqueeze(0)
            action_t, self.rnn_states = self.model(obs_t, self.rnn_states)
            # Clip come nel training
            action_t = torch.clamp(action_t, min=-1.0, max=1.0)
            raw_action = action_t.cpu().numpy().flatten()

        # 8. Update State
        self.prev_action = raw_action
        self.prev_ee_pos = current_ee_pos
        self.prev_ee_quat = current_ee_quat
        
        return raw_action
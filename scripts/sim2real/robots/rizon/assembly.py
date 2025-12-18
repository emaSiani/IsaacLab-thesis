        

import numpy as np
import torch
import yaml
import os
from controllers.policy_controller import PolicyController

class FlexivGearAssemblyPolicy(PolicyController):
    """
    Policy Controller per Flexiv Rizon 4s.
    Versione con Debug per NaNs.
    """

    def __init__(self) -> None:
        super().__init__()
        
        # 7 DOF del Rizon
        self.dof_names = [f"joint{i}" for i in range(1, 8)]
        self.num_joints = 7
        
        # deg: [-7,44, -32,7, 24,17, 102,3, -18,16, 40,4, 0.0]   
        self.default_pos = np.array([-0.13, -0.5707, 0.422, 1.7854, -0.317, 0.705, 0.0], dtype=np.float32)
        self.default_vel = np.zeros(self.num_joints, dtype=np.float32)

        # Path Policy
        self.policy_path = r"robots/rizon/policies/rizon4s_200ep_512envs_halved_orn_noise_policy.pt"
        self.config_path = r"robots/rizon/policies/rizon4s_200ep_512envs_halved_orn_noise_env_config.yaml"

        self._load_resources()

        self._action_scale = 0.5
        self._previous_action = np.zeros(self.num_joints)
        self._policy_counter = 0
        
        # Target Peg [x, y, z, qx, qy, qz, qw]
        self.peg_target_pose = np.array([0.5, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0]) 

        self.has_joint_data = False
        self.current_joint_positions = np.zeros(self.num_joints)
        self.current_joint_velocities = np.zeros(self.num_joints)
        self.gear_tip_position = np.zeros(3) 

    def _load_resources(self):
        print(f"[INFO] Loading policy from: {self.policy_path}")
        self.device = torch.device("cpu")
        
        try:
            self.model = torch.jit.load(self.policy_path, map_location=self.device)
            self.model.eval()
        except Exception as e:
            print(f"[ERROR] Cannot load JIT model: {e}")
            raise e

        try:
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)
            
            self._dt = cfg.get("dt", 0.0166)
            self._decimation = cfg.get("decimation", 2)
            self.num_obs = cfg.get("num_observations", 24) 
            self.rnn_units = cfg.get("rnn_units", 1024)
            self.rnn_layers = cfg.get("num_rnn_layers", 2)
            
            print(f"[INFO] Config Loaded. Obs: {self.num_obs}")
            
        except FileNotFoundError:
            print("[WARN] Config not found. Using defaults.")
            self._dt = 0.0166
            self._decimation = 2
            self.num_obs = 24
            self.rnn_units = 1024
            self.rnn_layers = 2

        # Init LSTM (h, c)
        h = torch.zeros((self.rnn_layers, 1, self.rnn_units), device=self.device)
        c = torch.zeros((self.rnn_layers, 1, self.rnn_units), device=self.device)
        self.rnn_states = (h, c)

    def update_joint_state(self, position, velocity) -> None:
        if len(position) >= self.num_joints:
            self.current_joint_positions = np.array(position[:self.num_joints], dtype=np.float32)
            self.current_joint_velocities = np.array(velocity[:self.num_joints], dtype=np.float32)
            self.has_joint_data = True

    def update_gear_tip_position(self, position) -> None:
        # Safety check for NaNs from TF
        if np.any(np.isnan(position)):
            print("[WARN] Received NaN in gear_tip_position! Ignoring update.")
            return
        self.gear_tip_position = np.array(position, dtype=np.float32)

    def _compute_observation(self, peg_pose: np.ndarray) -> np.ndarray:
        if not self.has_joint_data:
            return None
        
        obs = np.concatenate([
            self.current_joint_positions - self.default_pos[:7], 
            self.current_joint_velocities,                       
            self.gear_tip_position,                              
            peg_pose                                             
        ])
        
        if len(obs) != self.num_obs:
            obs_w_action = np.concatenate([obs, self._previous_action])
            if len(obs_w_action) == self.num_obs:
                return obs_w_action
                
        return obs

    def forward(self, dt: float) -> np.ndarray:
        if not self.has_joint_data:
            return None

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(self.peg_target_pose)
            if obs is None: return None

            # --- DEBUG NAN INPUTS ---
            if np.any(np.isnan(obs)):
                print(f"[CRITICAL] NaN detected in OBSERVATIONS!")
                print(f"Joint Pos: {self.current_joint_positions}")
                print(f"Gear Tip: {self.gear_tip_position}")
                return None # Skip step to avoid crash
            
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Inference
                action_t, self.rnn_states = self.model(obs_t, self.rnn_states)
                self.action = action_t.cpu().numpy().flatten()

                # --- DEBUG NAN OUTPUTS ---
                if np.any(np.isnan(self.action)):
                    print("[CRITICAL] Network produced NaN ACTION!")
                    # Reset action to zero to be safe
                    self.action = np.zeros(self.num_joints)
                
            self._previous_action = self.action.copy()

        # Calcolo Target
        joint_positions = self.default_pos[:7] + (self.action * self._action_scale)
        
        # Ultima sicurezza prima di ritornare
        if np.any(np.isnan(joint_positions)):
             print("[CRITICAL] joint_positions contains NaN! Resetting to current pos.")
             return self.current_joint_positions # Stay put
             
        self._policy_counter += 1
        return joint_positions
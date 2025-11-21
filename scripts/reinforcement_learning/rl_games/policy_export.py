import os
import torch
import yaml
from rl_games.common.player import BasePlayer

class DeployWrapper(torch.nn.Module):
    """
    Wraps the full RL-Games network to make it deploy-ready via JIT.
    Handles:
    1. Input Dictionary creation (so JIT accepts raw tensors).
    2. Full forward pass (Input Norm -> LSTM -> MLP).
    3. Output parsing (handles both Dict and Tuple returns).
    """
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, obs, rnn_states):
        """
        Args:
            obs: Raw observations [Batch, Num_Obs]
            rnn_states: Tuple of hidden states for LSTM
        Returns:
            action: Deterministic action
            new_rnn_states: Updated hidden states
        """
        # Construct the input dict expected by rl_games A2CNetwork
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': rnn_states
        }
        
        # Forward pass through the ENTIRE network
        result = self.network(input_dict)
        
        # --- ROBUST OUTPUT HANDLING ---
        # Check if result is a Dictionary (standard) or Tuple (some versions)
        if isinstance(result, dict):
            return result['mus'], result['rnn_states']
        
        # If it's a tuple, usually the format is: (mus, value, states) or (mus, states)
        # Index 0 is always the Action Mean (mus)
        # Index -1 is always the RNN States
        return result[0], result[-1]

def get_num_observations(agent):
    """Robustly detects observation size from various wrapper types."""
    # 1. Try direct attribute (RSL-RL style)
    if hasattr(agent, 'env') and hasattr(agent.env, 'num_observations'):
        return agent.env.num_observations
    
    # 2. Try observation_space on env (Gym/IsaacLab style)
    if hasattr(agent, 'env') and hasattr(agent.env, 'observation_space'):
        if hasattr(agent.env.observation_space, 'shape'):
            return agent.env.observation_space.shape[0]
            
    # 3. Try observation_space on agent (RL-Games BasePlayer)
    if hasattr(agent, 'observation_space') and hasattr(agent.observation_space, 'shape'):
        return agent.observation_space.shape[0]

    # 4. Fallback: Inspect the normalization layer if present
    if hasattr(agent.model, 'a2c_network') and hasattr(agent.model.a2c_network, 'running_mean_std'):
        return agent.model.a2c_network.running_mean_std.running_mean.shape[0]

    raise AttributeError("Could not detect 'num_observations' from agent or environment.")

def export_rl_games_policy(agent: BasePlayer, log_dir: str, task_name: str, rl_device: str):
    """
    Exports the RL-Games policy to TorchScript (JIT) and saves a config YAML.
    Ensures the network is returned to the original device after export.
    """
    network = agent.model.a2c_network
    
    try:
        print(f"\n[EXPORT] Starting Policy Export for {task_name}...")
        
        # 1. Move Network to CPU for Export (Standard practice for JIT portability)
        network.eval()
        network.cpu() 

        # 2. Detect Observation Size
        num_obs = get_num_observations(agent)
        print(f"[EXPORT] Detected Num Observations: {num_obs}")

        # 3. Prepare Dummy Inputs (Batch Size = 1)
        dummy_obs = torch.randn((1, num_obs), device="cpu")
        
        # 4. Prepare Dummy RNN States
        # Manual creation based on config (LSTM, 2 layers, 1024 units)
        # This prevents "TypeError" issues with get_default_rnn_state
        print("[EXPORT] Creating dummy RNN states (Layers=2, Units=1024)...")
        
        # LSTM needs hidden (h) and cell (c) states
        # Shape: (num_layers, batch_size, hidden_size)
        h = torch.zeros((2, 1, 1024), device="cpu")
        c = torch.zeros((2, 1, 1024), device="cpu")
        dummy_states = (h, c)

        # 5. Wrap and Trace
        deploy_model = DeployWrapper(network)
        
        export_dir = os.path.join(log_dir, "exported_policy")
        os.makedirs(export_dir, exist_ok=True)
        jit_path = os.path.join(export_dir, f"{task_name}_policy.pt")

        print("[EXPORT] Tracing model...")
        # We pass a tuple of arguments to trace
        traced_module = torch.jit.trace(deploy_model, (dummy_obs, dummy_states))
        traced_module.save(jit_path)

        print(f"[EXPORT] SUCCESS! Policy saved to: {jit_path}")
        print(f"[EXPORT] Model input signature: obs({num_obs}), rnn_states(LSTM Tuple)")

        # 6. Export Simple Config
        env_cfg_path = os.path.join(export_dir, f"{task_name}_env_config.yaml")
        env_data = {
            "dt": 0.0166, # Default 60Hz
            "decimation": 2, 
            "num_observations": num_obs,
            "num_rnn_layers": 2,   
            "rnn_units": 1024      
        }
        with open(env_cfg_path, 'w') as f:
            yaml.dump(env_data, f)
            
    except Exception as e:
        print(f"[EXPORT ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # === CRITICAL FIX ===
        # Always move the network back to the original device (GPU).
        # This prevents the simulator loop from crashing when it tries to use the network again.
        print(f"[EXPORT] Restoring network to device: {rl_device}")
        network.to(rl_device)
import os
import torch
import yaml
import numpy as np
from rl_games.common.player import BasePlayer

class DeployWrapper(torch.nn.Module):
    """
    Wraps the full RL-Games network AND the Normalization layer for JIT deployment.
    Forces everything to float32 to avoid JIT type errors.
    """
    def __init__(self, network, normalizer=None):
        super().__init__()
        self.network = network
        self.normalizer = normalizer
        self.normalize_input = (normalizer is not None)
        
        # Buffer per normalizzazione
        if self.normalize_input:
            self.register_buffer('running_mean', normalizer.running_mean.clone().float())
            self.register_buffer('running_var', normalizer.running_var.clone().float())
            # FIX: Epsilon come tensore float32 per evitare promozione a float64
            self.register_buffer('epsilon', torch.tensor(1e-05, dtype=torch.float32))

    def forward(self, obs, rnn_states):
        # 1. APPLICA NORMALIZZAZIONE
        # Assicuriamo casting esplicito all'ingresso
        obs = obs.float()
        
        if self.normalize_input:
            scale = torch.sqrt(self.running_var + self.epsilon)
            obs_norm = (obs - self.running_mean) / scale
            obs_norm = torch.clamp(obs_norm, -5.0, 5.0)
        else:
            obs_norm = obs

        # 2. Input Dict
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs_norm,
            'rnn_states': rnn_states
        }
        
        # 3. Forward Network
        result = self.network(input_dict)
        
        # 4. Output Parsing
        if isinstance(result, dict):
            return result['mus'], result['rnn_states']
        return result[0], result[-1]

def find_running_mean_std(agent):
    """Trova il layer di normalizzazione nell'agente."""
    if hasattr(agent.model, 'running_mean_std'):
        return agent.model.running_mean_std
    if hasattr(agent.model, 'a2c_network') and hasattr(agent.model.a2c_network, 'running_mean_std'):
        return agent.model.a2c_network.running_mean_std
    for name, module in agent.model.named_modules():
        if "RunningMeanStd" in str(type(module)):
            return module
    return None

def export_rl_games_policy(agent: BasePlayer, log_dir: str, task_name: str, rl_device: str, checkpoint_path: str = None):
    # 1. Carica Checkpoint
    if checkpoint_path:
        print(f"\n[EXPORT] Caricamento checkpoint: {checkpoint_path}")
        agent.restore(checkpoint_path)

    network = agent.model.a2c_network
    normalizer = find_running_mean_std(agent)
    
    # --- DIAGNOSTICA ---
    print("\n" + "="*50)
    print("       DIAGNOSTICA PRE-EXPORT")
    print("="*50)
    if normalizer is not None:
        mean = normalizer.running_mean.cpu().detach().numpy()
        print(f"✅ Normalizzatore TROVATO.")
        print(f"   Media (sample): {mean[:4]}")
    else:
        print("❌ Normalizzatore NON TROVATO.")
    print("="*50 + "\n")

    try:
        # 2. Preparazione CPU e CASTING FLOAT32 (Fix Critico)
        network.eval()
        network.cpu()
        network.float() # <--- FORZA PESI RETE A FLOAT32
        
        if normalizer:
            normalizer.eval()
            normalizer.cpu()
            normalizer.float() # <--- FORZA NORMALIZZATORE A FLOAT32

        # Rilevamento dimensioni
        if hasattr(agent.env, 'num_observations'):
            num_obs = agent.env.num_observations
        elif hasattr(agent.env, 'observation_space'):
            num_obs = agent.env.observation_space.shape[0]
        elif normalizer is not None:
             num_obs = normalizer.running_mean.shape[0]
        else:
             num_obs = 24 

        print(f"[EXPORT] Num Observations: {num_obs}")

        # 3. Dummy Input (Tutto Float32)
        dummy_obs = torch.randn((1, num_obs), device="cpu").float()
        h = torch.zeros((2, 1, 1024), device="cpu").float()
        c = torch.zeros((2, 1, 1024), device="cpu").float()
        dummy_states = (h, c)

        # 4. Tracing
        deploy_model = DeployWrapper(network, normalizer)
        # Forza anche il wrapper (per sicurezza sui buffer)
        deploy_model.float() 
        
        export_dir = os.path.join(log_dir, "exported_policy")
        os.makedirs(export_dir, exist_ok=True)
        jit_path = os.path.join(export_dir, f"{task_name}_policy.pt")

        print("[EXPORT] Tracing JIT model...")
        traced_module = torch.jit.trace(deploy_model, (dummy_obs, dummy_states))
        traced_module.save(jit_path)
        print(f"✅ [EXPORT] Policy salvata: {jit_path}")

        # 5. Config YAML
        env_cfg_path = os.path.join(export_dir, f"{task_name}_env_config.yaml")
        env_data = {
            "dt": 0.0166,
            "decimation": 2, 
            "num_observations": num_obs,
            "num_rnn_layers": 2,   
            "rnn_units": 1024       
        }
        with open(env_cfg_path, 'w') as f:
            yaml.dump(env_data, f)

    except Exception as e:
        print(f"❌ [EXPORT ERROR]: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Ripristino
        network.to(rl_device)
        if normalizer:
            normalizer.to(rl_device)
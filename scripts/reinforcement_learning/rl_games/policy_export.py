import os
import torch
from rl_games.common.player import BasePlayer
from typing import Dict, Any

# Classe wrapper per l'esportazione: combina i layer di policy
# Assumiamo che la policy sia la catena actor_mlp -> mu
class ActorMLP(torch.nn.Module):
    """
    Modulo PyTorch che incapsula la parte Attore (MLP e layer mu) della rete RL-Games.
    Questo e' necessario per esportare il forward pass della policy separato 
    dalle componenti critico e RNN.
    """
    def __init__(self, actor_mlp, mu_layer):
        super().__init__()
        self.actor_mlp = actor_mlp
        self.mu_layer = mu_layer
        
    def forward(self, x):
        # x e' l'input all'MLP (lo stato latente di 1024 nel tuo caso RNN)
        features = self.actor_mlp(x)
        action_mean = self.mu_layer(features)
        return action_mean


def export_rl_games_policy(agent: BasePlayer, log_dir: str, task_name: str, rl_device: str):
    """
    Esporta la policy (policy_model) da un agente RL-Games in formato ONNX e TorchScript JIT (.jit/.pt).
    
    ATTENZIONE: Questa funzione assume che la policy sia una rete ricorrente (RNN)
    e che l'input al policy_model sia lo stato latente dell'RNN (dimensione 1024).

    Args:
        agent: L'agente BasePlayer di RL-Games caricato.
        log_dir: La directory radice dove salvare l'esportazione.
        task_name: Il nome del task per nominare i file.
        rl_device: Il dispositivo (es. 'cuda:0') su cui risiede il modello.
    """
    try:
        # Mettere il modello in modalita' inferenza
        agent.model.a2c_network.eval()
        
        # Estrarre i layer della policy
        actor_mlp = agent.model.a2c_network.actor_mlp
        mu_layer = agent.model.a2c_network.mu
        
        policy_model = ActorMLP(actor_mlp, mu_layer)

        # Creare un Input Fittizio Corretto (dimensione 1024 per l'actor_mlp)
        # 1024 e' la dimensione di input per actor_mlp come visto dalla tua configurazione
        INPUT_SIZE_RNN_FEATURES = 1024
        dummy_input = torch.randn(
            (1, INPUT_SIZE_RNN_FEATURES),  # [Batch_size=1, Dimensione Feature RNN]
            device=rl_device
        )
        
        export_dir = os.path.join(log_dir, "exported_policy")
        os.makedirs(export_dir, exist_ok=True)
        
        print("\n" + "="*50)
        print("INIZIO ESPORTAZIONE POLICY RL-GAMES (RNN)")
        print(f"Directory di output: {export_dir}")
        print(f"ATTENZIONE: Policy esportata richiede input di dimensione {INPUT_SIZE_RNN_FEATURES}.")
        print("="*50)
        
        # 1. Esportazione in formato TorchScript (.pt / .jit)
        # rsl_rl usa .pt, rl_games .jit, ma sono entrambi TorchScript
        jit_path = os.path.join(export_dir, f"{task_name}_policy.pt")
        print(f"[INFO] Esportazione TorchScript (.pt): {jit_path}")
        traced_script_module = torch.jit.trace(policy_model, dummy_input)
        traced_script_module.save(jit_path)

        # 2. Esportazione in formato ONNX
        onnx_path = os.path.join(export_dir, f"{task_name}_policy.onnx")
        print(f"[INFO] Esportazione ONNX (.onnx): {onnx_path}")
        torch.onnx.export(
            policy_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["rnn_features"],
            output_names=["action_mean"],
            dynamic_axes={"rnn_features": {0: "batch_size"}, "action_mean": {0: "batch_size"}},
        )
        
        print("[INFO] Esportazione policy completata con successo.")
        
    except AttributeError as e:
        print(f"\n[ERRORE ESPORTAZIONE] Impossibile trovare i layer di policy (actor_mlp o mu): {e}")
        print("Skipping policy export.")
    except Exception as e:
        print(f"\n[ERRORE GENERICO ESPORTAZIONE] Si e' verificato un errore durante l'esportazione: {e}")
        print("Skipping policy export.")


def export_environment_config(env_cfg: Dict[str, Any], log_dir: str, task_name: str):
    """
    Esporta la configurazione essenziale dell'ambiente in un file YAML.
    Questo simula l'esportazione dell'environment usata in rsl_rl.

    Args:
        env_cfg: La configurazione dell'ambiente (oggetto ManagerBasedRLEnvCfg, etc.).
        log_dir: La directory radice dove salvare l'esportazione.
        task_name: Il nome del task per nominare il file.
    """
    import yaml
    from isaaclab.envs import DirectRLEnvCfg

    export_dir = os.path.join(log_dir, "exported_policy")
    os.makedirs(export_dir, exist_ok=True)
    yaml_path = os.path.join(export_dir, f"{task_name}_env_config.yaml")

    # Estraiamo i dati essenziali: osservazioni, azioni e dt
    if isinstance(env_cfg, DirectRLEnvCfg):
        env_data = {
            "num_observations": env_cfg.num_observations,
            "num_actions": env_cfg.num_actions,
            "dt": env_cfg.sim.dt,
            "clip_actions": True, # Assumiamo clipping delle azioni per coerenza
            # Aggiungi qui altri parametri rilevanti se necessari
        }
    else:
        # Per altri tipi di env (MARL o ManagerBased), potresti dover accedere diversamente
        # Usiamo l'approccio generico se disponibile o passiamo oltre
        env_data = {
            "num_observations": "N/A (Verificare env.observation_space.shape)",
            "num_actions": "N/A (Verificare env.action_space.shape)",
            "dt": env_cfg.sim.dt,
        }
    
    # Scriviamo il file YAML
    try:
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(env_data, f, sort_keys=False)
        print(f"[INFO] Esportazione configurazione Environment (.yaml) completata: {yaml_path}")
    except Exception as e:
        print(f"[ERRORE ESPORTAZIONE YAML] Impossibile scrivere il file YAML: {e}")
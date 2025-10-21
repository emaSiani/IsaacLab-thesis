# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .rizon4s_forge_env import Rizon4sForgeEnv
from .rizon4s_forge_env_cfg import Rizon4sForgeTaskGearMeshCfg, Rizon4sForgeTaskNutThreadCfg, Rizon4sForgeTaskPegInsertCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Rizon4s-Forge-PegInsert-Direct-v0",
    entry_point="isaaclab_tasks.direct.rizon4s_forge:Rizon4sForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Rizon4sForgeTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Rizon4s-Forge-GearMesh-Direct-v0",
    entry_point="isaaclab_tasks.direct.rizon4s_forge:Rizon4sForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Rizon4sForgeTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Rizon4s-Forge-NutThread-Direct-v0",
    entry_point="isaaclab_tasks.direct.rizon4s_forge:Rizon4sForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Rizon4sForgeTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)

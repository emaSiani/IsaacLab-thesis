# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .rizon4s_factory_env import Rizon4sFactoryEnv
from .rizon4s_factory_env_cfg import Rizon4sFactoryTaskGearMeshCfg, Rizon4sFactoryTaskNutThreadCfg, Rizon4sFactoryTaskPegInsertCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Rizon4s-Factory-PegInsert-Direct-v0",
    entry_point="isaaclab_tasks.direct.rizon4s_factory:Rizon4sFactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Rizon4sFactoryTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Rizon4s-Factory-GearMesh-Direct-v0",
    entry_point="isaaclab_tasks.direct.rizon4s_factory:Rizon4sFactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Rizon4sFactoryTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Rizon4s-Factory-NutThread-Direct-v0",
    entry_point="isaaclab_tasks.direct.rizon4s_factory:Rizon4sFactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Rizon4sFactoryTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

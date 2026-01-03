"""Package containing task implementations for various robotic environments."""

import os
import toml

from isaaclab_tasks.utils import import_packages
import gymnasium as gym

##
# Register Gym environments.
##

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

gym.register(
    id="Isaac-Limx-PF-Stunt-OneLeg-v0",  # 任务ID (train.py --task 参数用这个)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 👇 关键：指向你刚才新建的配置文件
        # 对应路径: .../tasks/locomotion/cfg/PF/stunt_one_leg_env_cfg.py
        "env_cfg_entry_point": "bipedal_locomotion.tasks.locomotion.cfg.PF.stunt_one_leg_env_cfg:PFEnvCfg",

        # 👇 PPO配置：复用你现有的平地PPO配置即可 (路径请根据你实际情况确认)
        # 假设你的 rsl_rl_ppo_cfg.py 在 agents 目录下
        "rsl_rl_cfg_entry_point": "bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg:PF_TRON1AFlatPPORunnerCfg",
    },
)

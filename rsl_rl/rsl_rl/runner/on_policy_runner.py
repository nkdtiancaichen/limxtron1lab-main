# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from rsl_rl.algorithm import PPO
from rsl_rl.modules import MLP_Encoder, ActorCritic
from rsl_rl.env import VecEnv


class OnPolicyRunner:
    """在线策略训练器 - 管理PPO算法的完整训练过程 / On-policy trainer - manages complete training process for PPO algorithm"""
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        """初始化训练器 / Initialize trainer

        Args:
            env: 向量化环境 / Vectorized environment
            train_cfg: 训练配置 / Training configuration
            log_dir: 日志目录 / Log directory
            device: 计算设备 / Computing device
        """
        self.cfg = train_cfg
        print(f"encoder cfg: {train_cfg.keys()}")
        self.ecd_cfg = train_cfg["encoder"]        # 编码器配置 / Encoder configuration
        self.alg_cfg = train_cfg["algorithm"]      # 算法配置 / Algorithm configuration
        self.policy_cfg = train_cfg["policy"]      # 策略配置 / Policy configuration
        self.device = device
        self.env = env

        # 获取环境观测信息 / Get environment observation information
        # -------- 针对 TensorDict 的最终修复 --------
        returns = self.env.get_observations()
        ret_type = str(type(returns))  # 获取类型字符串，避免导入报错
        print(f"DEBUG: get_observations return type: {ret_type}")

        # [情况 A] 新版 Isaac Lab 返回 TensorDict (增强版)
        if str(type(returns)).find("TensorDict") != -1 or isinstance(returns, dict):
            # 打印一下键值，方便调试 (看看是叫 'obsHistory' 还是 'obs_history')
            # print(f"DEBUG Keys: {returns.keys()}")

            # 1. 提取策略观测值
            if "policy" in returns.keys():
                obs = returns["policy"]
            elif "obs" in returns.keys():
                obs = returns["obs"]
            else:
                obs = returns

                # 2. 提取 extras (必须包含 commands 和 obsHistory)
            extras = {"observations": {}}

            # --- 辅助函数：不区分大小写/格式查找键值 ---
            def find_key(target_dict, key_candidates):
                for k in key_candidates:
                    if k in target_dict.keys():
                        return target_dict[k]
                return None

            # ---------------------------------------

            # 提取 commands
            cmd_data = find_key(returns, ["commands"])
            # 如果顶层没有，去 observations 子层找
            if cmd_data is None and "observations" in returns.keys():
                cmd_data = find_key(returns["observations"], ["commands"])

            if cmd_data is not None:
                extras["observations"]["commands"] = cmd_data

            # 提取 obsHistory (关键修复！！！)
            # Isaac Lab 可能会把 camelCase 转为 snake_case，所以我们都试一下
            hist_data = find_key(returns, ["obsHistory", "obs_history", "history"])
            if hist_data is not None:
                extras["observations"]["obsHistory"] = hist_data
            else:
                print("[WARNING] 'obsHistory' not found in TensorDict! Encoder might fail.")

            # 提取 critic
            crit_data = find_key(returns, ["critic"])
            if crit_data is not None:
                extras["observations"]["critic"] = crit_data

        # [情况 B] 旧版兼容 (Tuple)
        elif isinstance(returns, tuple):
            if len(returns) == 3:
                obs, privileged_obs, extras = returns
            else:
                obs, extras = returns
                privileged_obs = None

        # [情况 C] 原始 Tensor (你上一次遇到的情况)
        elif torch.is_tensor(returns):
            obs = returns
            privileged_obs = None
            # 尝试手动补救 extras
            if hasattr(self.env, "extras"):
                extras = self.env.extras
            else:
                extras = {"observations": {}}

        else:
            raise TypeError(f"Unknown return type: {type(returns)}")

        # 🚨 再次检查：确保 commands 存在 🚨
        if "observations" not in extras or "commands" not in extras["observations"]:
            print("[WARNING] 'commands' still missing. Trying env.unwrapped.command_manager...")
            try:
                base_env = self.env.unwrapped
                if hasattr(base_env, "command_manager"):
                    cmds = base_env.command_manager.get_command(None)
                    if "observations" not in extras: extras["observations"] = {}
                    extras["observations"]["commands"] = cmds
            except Exception as e:
                print(f"[ERROR] Failed to force load commands: {e}")
        # -----------------------------------------------
        self.num_obs = obs.shape[1]
        self.obs_history_len = self.alg_cfg.pop("obs_history_len")

        # 验证必要的观测组 / Verify necessary observation groups
        assert "commands" in extras["observations"], f"Commands not found in observations"
        self.num_commands = extras["observations"]["commands"].shape[1]
        assert "critic" in extras["observations"], f"Critic observations not found in observations"
        num_critic_obs = extras["observations"]["critic"].shape[1] + self.num_commands

        # 设置编码器输入维度 / Set encoder input dimensions
        self.ecd_cfg["num_input_dim"] = self.obs_history_len * self.num_obs

        # 创建编码器 / Create encoder
        encoder = eval("MLP_Encoder")(
            **self.ecd_cfg,
        ).to(self.device)

        # 创建Actor-Critic网络 / Create Actor-Critic network
        actor_critic_class = eval("ActorCritic")  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            self.num_obs + encoder.num_output_dim + self.num_commands,  # Actor输入维度 / Actor input dimensions
            num_critic_obs,                                             # Critic输入维度 / Critic input dimensions
            self.env.num_actions,                                       # 动作维度 / Action dimensions
            **self.policy_cfg,
        ).to(self.device)

        # 创建PPO算法实例 / Create PPO algorithm instance
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg = alg_class(
            self.env.num_envs,
            encoder,
            actor_critic,
            device = self.device,
            **self.alg_cfg,
        )

        # 训练参数 / Training parameters
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # 初始化存储和模型 / Initialize storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.num_obs],
            [num_critic_obs],
            [self.obs_history_len * self.num_obs],
            [self.num_commands],
            [self.env.num_actions],
        )

        # 观测标准化参数 / Observation normalization parameters
        self.obs_mean = torch.tensor(
            0, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.obs_std = torch.tensor(
            1, dtype=torch.float, device=self.device, requires_grad=False
        )

        # 日志设置 / Logging setup
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # 重置环境 / Reset environment
        _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """执行学习过程 / Execute learning process

        Args:
            num_learning_iterations: 学习迭代次数 / Number of learning iterations
            init_at_random_ep_len: 是否从随机episode长度开始 / Whether to start from random episode length
        """
        # 初始化日志记录器 / Initialize logger
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Wandb & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "wandb":
                from ..utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "tensorboard":
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        # 随机化episode长度 / Randomize episode length
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 获取初始观测 / Get initial observations
            # -------- 修复 learn 函数开头的 get_observations --------
            returns = self.env.get_observations()

            # [情况 A] 新版 Isaac Lab 返回 TensorDict (增强版)
            if str(type(returns)).find("TensorDict") != -1 or isinstance(returns, dict):
                # 打印一下键值，方便调试 (看看是叫 'obsHistory' 还是 'obs_history')
                # print(f"DEBUG Keys: {returns.keys()}")

                # 1. 提取策略观测值
                if "policy" in returns.keys():
                    obs = returns["policy"]
                elif "obs" in returns.keys():
                    obs = returns["obs"]
                else:
                    obs = returns

                    # 2. 提取 extras (必须包含 commands 和 obsHistory)
                extras = {"observations": {}}

                # --- 辅助函数：不区分大小写/格式查找键值 ---
                def find_key(target_dict, key_candidates):
                    for k in key_candidates:
                        if k in target_dict.keys():
                            return target_dict[k]
                    return None

                # ---------------------------------------

                # 提取 commands
                cmd_data = find_key(returns, ["commands"])
                # 如果顶层没有，去 observations 子层找
                if cmd_data is None and "observations" in returns.keys():
                    cmd_data = find_key(returns["observations"], ["commands"])

                if cmd_data is not None:
                    extras["observations"]["commands"] = cmd_data

                # 提取 obsHistory (关键修复！！！)
                # Isaac Lab 可能会把 camelCase 转为 snake_case，所以我们都试一下
                hist_data = find_key(returns, ["obsHistory", "obs_history", "history"])
                if hist_data is not None:
                    extras["observations"]["obsHistory"] = hist_data
                else:
                    print("[WARNING] 'obsHistory' not found in TensorDict! Encoder might fail.")

                # 提取 critic
                crit_data = find_key(returns, ["critic"])
                if crit_data is not None:
                    extras["observations"]["critic"] = crit_data

            # [情况 B] 旧版兼容 (Tuple)
            elif isinstance(returns, tuple):
                if len(returns) == 3:
                    obs, privileged_obs, extras = returns
                else:
                    obs, extras = returns

            # [情况 C] 原始 Tensor
            elif torch.is_tensor(returns):
                obs = returns
                # 尝试补救 extras
                if hasattr(self.env, "extras"):
                    extras = self.env.extras
                else:
                    extras = {"observations": {}}
            # ------------------------------------------------------
        obs_history = extras["observations"].get("obsHistory")
        obs_history = obs_history.flatten(start_dim=1)
        critic_obs = extras["observations"].get("critic")
        commands = extras["observations"].get("commands")

        obs, obs_history, commands, critic_obs = (
            obs.to(self.device),
            obs_history.to(self.device),
            commands.to(self.device),
            critic_obs.to(self.device),
        )

        # 切换到训练模式 / Switch to train mode
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        # 训练统计 / Training statistics
        ep_infos = []
        rewbuffer = deque(maxlen=100)    # 奖励缓冲区 / Reward buffer
        lenbuffer = deque(maxlen=100)    # 长度缓冲区 / Length buffer
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # 数据收集阶段 / Data collection phase
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    # 执行动作 / Execute actions
                    actions = self.alg.act(obs, obs_history, commands, critic_obs)

                    # 环境步进 / Environment step
                    step_returns = self.env.step(actions)

                    # --- [关键修复开始] 数据解包与填补 ---
                    # 1. 基础解包
                    if len(step_returns) == 5:
                        obs, privileged_obs, rewards, dones, infos = step_returns
                    elif len(step_returns) == 4:
                        obs, rewards, dones, infos = step_returns
                        privileged_obs = None
                    else:
                        # 极端保底
                        obs, rewards, dones, infos = step_returns[0], step_returns[1], step_returns[2], step_returns[3]
                        privileged_obs = None

                    # 2. 确保 infos 结构完整
                    if "observations" not in infos:
                        infos["observations"] = {}

                    # 3. 处理 TensorDict (如果 obs 是个大字典，拆分它)
                    if isinstance(obs, dict) or "TensorDict" in str(type(obs)):
                        full_dict = obs

                        # A. 提取 Policy Obs (覆盖 obs 变量，只保留策略部分)
                        if "policy" in full_dict.keys():
                            obs = full_dict["policy"]
                        elif "obs" in full_dict.keys():
                            obs = full_dict["obs"]

                        # B. 提取 Critic Obs -> 存入 infos
                        if "critic" in full_dict.keys():
                            infos["observations"]["critic"] = full_dict["critic"]

                        # C. 提取 Obs History -> 存入 infos
                        # 尝试多种可能的键名
                        for k in ["obsHistory", "obs_history", "history"]:
                            if k in full_dict.keys():
                                infos["observations"]["obsHistory"] = full_dict[k]
                                break

                        # D. 提取 Commands -> 存入 infos
                        if "commands" in full_dict.keys():
                            infos["observations"]["commands"] = full_dict["commands"]

                    # 4. [最后的安全网] 防止 KeyError
                    # 如果上面还没提取到，尝试用 obs 或 零矩阵填充，防止崩溃
                    if "critic" not in infos["observations"]:
                        # print("[WARN] Missing 'critic', using obs as fallback")
                        infos["observations"]["critic"] = obs  # 临时替补

                    if "obsHistory" not in infos["observations"]:
                        # print("[WARN] Missing 'obsHistory', using obs as fallback")
                        infos["observations"]["obsHistory"] = obs  # 临时替补 (维度可能不对，但在 reset 后通常有历史)

                    if "commands" not in infos["observations"]:
                        # print("[WARN] Missing 'commands', attempting fetch")
                        try:
                            # 尝试从环境直接抓取
                            if hasattr(self.env.unwrapped, "command_manager"):
                                infos["observations"]["commands"] = self.env.unwrapped.command_manager.get_command(None)
                            else:
                                infos["observations"]["commands"] = torch.zeros((self.env.num_envs, self.num_commands),
                                                                                device=self.device)
                        except:
                            infos["observations"]["commands"] = torch.zeros((self.env.num_envs, self.num_commands),
                                                                            device=self.device)
                    # --- [关键修复结束] ---

                    # 更新观测 / Update observations
                    # 现在这里的读取应该是安全的了
                    critic_obs = infos["observations"]["critic"]
                    obs_history = infos["observations"]["obsHistory"].flatten(start_dim=1)
                    commands = infos["observations"]["commands"]

                    # 转换到设备 / Transfer to device
                    obs, obs_history, commands, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        obs_history.to(self.device),
                        commands.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    # 处理环境步进结果 / Process environment step results
                    self.alg.process_env_step(rewards, dones, infos, obs)

                    if self.log_dir is not None:
                        # 记录统计信息 / Record statistics
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # 学习阶段 / Learning phase
                start = stop

                # 准备评价器观测 / Prepare critic observations
                critic_obs_ = torch.cat((critic_obs, commands), dim=-1)
                if self.alg.critic_take_latent:
                    encoder_out = self.alg.encoder.encode(obs_history)
                    self.alg.compute_returns(
                        torch.cat((critic_obs_, encoder_out), dim=-1)
                    )
                else:
                    self.alg.compute_returns(critic_obs_)

            # 执行PPO更新 / Perform PPO update
            (
                mean_value_loss,
                mean_extra_loss,
                mean_surrogate_loss,
                mean_kl,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )

    def log(self, locs, width=80, pad=35):
        """记录训练统计信息 / Log training statistics

        Args:
            locs: 局部变量字典 / Local variables dictionary
            width: 控制台输出宽度 / Console output width
            pad: 填充长度 / Padding length
        """
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
        # mean_std = self.alg.actor_critic.std.mean()
        mean_std = torch.exp(self.alg.actor_critic.logstd).mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/encoder", locs["mean_extra_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Policy/mean_kl", locs["mean_kl"], locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            if (
                self.logger_type != "wandb"
            ):  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.4f}\n"""
                f"""{'Learning rate:':>{pad}} {self.alg.learning_rate:.4f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "encoder_state_dict": self.alg.encoder.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=False):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.encoder.load_state_dict(loaded_dict["encoder_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_encoder(self, device=None):
        self.alg.encoder.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.encoder.to(device)
        return self.alg.encoder.encode

    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic

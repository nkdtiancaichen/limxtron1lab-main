"""RSL-RL智能体检查点播放脚本 / Script to play a checkpoint of an RL agent from RSL-RL."""

"""首先启动Isaac Sim仿真器 / Launch Isaac Sim Simulator first."""

import argparse
import copy  # [新增] 用于复制神经网络策略

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 添加argparse参数 / Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
parser.add_argument("--hop_checkpoint_path", type=str, default=None, help="Path to the hopping/stunt model checkpoint file.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit

# 在文件最上方添加
import sys
from pynput import keyboard
import threading


# --- 键盘控制器类 ---
# --- 键盘控制器类 (完整修正版) ---
class KeyboardInterface:
    def __init__(self):
        self.vx = 0.0  # 前后速度
        self.vy = 0.0  # 左右平移
        self.wz = 0.0  # 旋转速度

        # 👇👇👇 [关键修正] 必须在这里初始化这个变量！
        self.use_hopping_policy = False

        # 启动监听线程
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        print("\n" + "=" * 30)
        print("  键盘控制已激活 (Keyboard Active)")
        print("  W / S : 前进 / 后退 (Vx)")
        print("  A / D : 左移 / 右移 (Vy)")
        print("  Q / E : 左转 / 右转 (Yaw)")
        print("  Space : 急停 (Stop)")
        print("  K     : 切换行走/单脚跳模式 (Toggle Mode)")  # [新增] 提示
        print("=" * 30 + "\n")

    def on_press(self, key):
        try:
            # 步长设置
            step_lin = 0.1
            step_ang = 0.1

            if hasattr(key, 'char'):
                if key.char == 'w':
                    self.vx += step_lin
                elif key.char == 's':
                    self.vx -= step_lin
                elif key.char == 'a':
                    self.vy += step_lin
                elif key.char == 'd':
                    self.vy -= step_lin
                elif key.char == 'q':
                    self.wz += step_ang
                elif key.char == 'e':
                    self.wz -= step_ang
                elif key.char == ' ':
                    self.vx, self.vy, self.wz = 0, 0, 0

                # 👇👇👇 [关键修正] K键切换逻辑
                elif key.char == 'k':
                    self.use_hopping_policy = not self.use_hopping_policy
                    mode_str = "【单脚跳 HOPPING】" if self.use_hopping_policy else "【正常行走 WALKING】"
                    # 加 \r 确保打印不乱行
                    print(f"\r>>> 切换模式: {mode_str}                  ")

            # 限幅
            self.vx = max(min(self.vx, 1.5), -1.0)
            self.vy = max(min(self.vy, 0.5), -0.5)
            self.wz = max(min(self.wz, 1.5), -1.5)

            # 实时打印当前指令
            sys.stdout.write(f"\r[Cmd] Vx: {self.vx:.2f} | Vy: {self.vy:.2f} | Wz: {self.wz:.2f}   ")
            sys.stdout.flush()

        except AttributeError:
            pass


# --------------------

def main():
    """使用RSL-RL智能体进行测试 / Play with RSL-RL agent."""

    # 1. 在这里初始化键盘控制器 (在 gym.make 或 env创建前后均可)
    keyboard_cmd = KeyboardInterface()

    # 解析配置 / Parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    # 指定日志实验目录 / Specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # 创建isaac环境 / Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    walk_checkpoint = resume_path
    print(f"[INFO]: Loading WALKING model from: {walk_checkpoint}")
    ppo_runner.load(walk_checkpoint)

    # 深拷贝保存行走策略
    policy_walk = copy.deepcopy(ppo_runner.get_inference_policy(device=env.unwrapped.device))
    encoder_walk = copy.deepcopy(ppo_runner.get_inference_encoder(device=env.unwrapped.device))

    hop_checkpoint = args_cli.hop_checkpoint_path
    if hop_checkpoint and os.path.exists(hop_checkpoint):
        print(f"\n[INFO]: Loading HOPPING model from: {hop_checkpoint}")
        # 加载单脚跳权重
        ppo_runner.load(hop_checkpoint)

        # 深拷贝保存单脚跳策略
        policy_hop = copy.deepcopy(ppo_runner.get_inference_policy(device=env.unwrapped.device))
        encoder_hop = copy.deepcopy(ppo_runner.get_inference_encoder(device=env.unwrapped.device))
        print("[SUCCESS] Hopping policy loaded successfully. Press 'K' to toggle.\n")
    else:
        if hop_checkpoint:
            print(f"[WARNING]: 指定的单脚跳模型路径不存在: {hop_checkpoint}")
        else:
            print("[INFO]: 未指定单脚跳模型 (--hop_checkpoint_path)，K键切换将无效。")

        # 如果没传参数或路径不对，默认回退到行走策略，防止报错
        policy_hop = policy_walk
        encoder_hop = encoder_walk

     # 导出策略到onnx / Export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor, 
            export_model_dir, 
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )
    # reset environment
    # -------- 修复 1: 初始化 get_observations (构造嵌套结构) --------
    returns = env.get_observations()

    # 动态处理 TensorDict
    if str(type(returns)).find("TensorDict") != -1 or isinstance(returns, dict):
        # 1. 提取 Policy 输入 (obs)
        if "policy" in returns.keys():
            obs = returns["policy"]
        elif "obs" in returns.keys():
            obs = returns["obs"]
        else:
            obs = returns

        # 2. 构造旧代码期望的嵌套字典 {"observations": {...}}
        # 这一步是解决 KeyError 的关键！
        obs_dict = {"observations": {}}

        # 自动搬运所有键值到 observations 下
        # 尤其是 obsHistory, commands, critic
        for key in returns.keys():
            # 统一处理键名 (比如把 obs_history 映射回 obsHistory)
            if key == "obs_history":
                obs_dict["observations"]["obsHistory"] = returns[key]
            else:
                obs_dict["observations"][key] = returns[key]

    elif isinstance(returns, tuple):
        # 旧版兼容
        if len(returns) == 3:
            obs, privileged_obs, obs_dict = returns
        else:
            obs, obs_dict = returns
    else:
        obs = returns
        obs_dict = {"observations": {}}
    # -----------------------------------------------------------
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands") 
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # 1. 👇 先决定用哪个模型 (这块代码你可能已经加了)
            if keyboard_cmd.use_hopping_policy:
                current_policy = policy_hop
                current_encoder = encoder_hop
            else:
                current_policy = policy_walk
                current_encoder = encoder_walk

            # agent stepping
            est = current_encoder(obs_history)
            actions = current_policy(torch.cat((est, obs, commands), dim=-1).detach())
            # env stepping
            # -------- 最终修复: 变量名修正为 infos --------
            step_returns = env.step(actions)

            # 1. 解包 (变量名必须叫 infos，配合后面的代码)
            if len(step_returns) == 5:
                # Isaac Lab 0.47+: obs, privileged_obs, rewards, dones, infos
                obs, privileged_obs, rewards, dones, infos = step_returns
            elif len(step_returns) == 4:
                # 旧版标准: obs, rewards, dones, infos
                obs, rewards, dones, infos = step_returns
            else:
                obs = step_returns[0]
                infos = {}  # 防止崩溃

            # 2. 处理 TensorDict -> 填充 infos
            if str(type(obs)).find("TensorDict") != -1 or isinstance(obs, dict):
                full_dict = obs

                # A. 提取 Policy 输入 (真正喂给神经网络的 obs)
                if "policy" in full_dict.keys():
                    obs = full_dict["policy"]
                elif "obs" in full_dict.keys():
                    obs = full_dict["obs"]

                # B. 搬运数据到 infos (因为后面的代码从 infos 里取 obsHistory)
                # 确保 infos["observations"] 存在
                if "observations" not in infos:
                    infos["observations"] = {}

                # 将 TensorDict 里的所有数据（如 obsHistory, commands）搬运进去
                for key in full_dict.keys():
                    if key == "obs_history":
                        infos["observations"]["obsHistory"] = full_dict[key]
                    else:
                        infos["observations"][key] = full_dict[key]

            # 3. 保底检查
            if "observations" not in infos:
                infos["observations"] = {}
            # -----------------------------------------------------
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = infos["observations"].get("commands")

            # ============ 【关键修改】强行覆盖 Commands ============
            # ====================================================
            if commands is not None:
                # commands 的形状通常是 [num_envs, 3] -> (vx, vy, wz)
                # 无论有多少个环境，我们把所有机器人的指令都设为键盘控制的值

                # 覆盖 X 轴线速度 (Vx)
                commands[:, 0] = keyboard_cmd.vx

                # 覆盖 Y 轴线速度 (Vy)
                commands[:, 1] = keyboard_cmd.vy

                # 覆盖 Z 轴角速度 (Yaw)
                commands[:, 2] = keyboard_cmd.wz

                # (可选) 如果你的指令包含更多维度 (如高度、频率)，保持原样或手动设置
                # commands[:, 3] = ...
            else:
                # 如果环境甚至没返回 commands，我们手动造一个
                # 这在某些极端情况下是必要的防崩溃措施
                commands = torch.zeros((env.num_envs, 3), device=env.device)
                commands[:, 0] = keyboard_cmd.vx
                commands[:, 1] = keyboard_cmd.vy
                commands[:, 2] = keyboard_cmd.wz
            # ====================================================

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()

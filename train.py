# train.py
import os
from typing import List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import Config
from env import FASEnv


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.running = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)

        if rewards is not None:
            self.running += float(np.mean(rewards))
        if dones is not None and np.any(dones):
            self.episode_rewards.append(self.running)
            self.running = 0.0
        return True


def make_env(cfg: Config):
    def _init():
        env = FASEnv(cfg)
        return Monitor(env)

    return _init


def run_experiment(exp_name: str, cfg: Config):
    print(f"\n{'=' * 20} Starting Experiment: {exp_name} {'=' * 20}")
    print(f"Memetic Search: {cfg.use_memetic_search}")

    vec_env = DummyVecEnv([make_env(cfg)])

    # 稍微增大网络容量以处理增加了 Mask 的观测空间
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        policy_kwargs=policy_kwargs,
        seed=cfg.seed,
        verbose=1,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        tensorboard_log=f"./tb_logs/{exp_name}"
    )

    cb = RewardLoggerCallback()
    model.learn(total_timesteps=cfg.total_timesteps, callback=cb)

    # 保存模型和奖励
    model_dir = os.path.join(cfg.out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, f"ppo_{exp_name}"))

    reward_file = os.path.join(cfg.out_dir, f"train_rewards_{exp_name}.npy")
    np.save(reward_file, np.array(cb.episode_rewards, dtype=float))
    print(f"Saved rewards to {reward_file}")
    return cb.episode_rewards


def plot_results(rewards_dict: dict, save_path: str):
    plt.figure(figsize=(10, 6))
    for label, rewards in rewards_dict.items():
        # 简单的平滑处理
        if len(rewards) > 0:
            rewards = np.array(rewards)
            # 使用滑动窗口平滑，窗口大小为 10% 的回合数或至少 5
            window = max(5, len(rewards) // 20)
            if len(rewards) >= window:
                rewards_padded = np.pad(rewards, (window // 2, window // 2), mode='edge')
                rewards_smooth = np.convolve(rewards_padded, np.ones(window) / window, mode='valid')
            else:
                rewards_smooth = rewards

            plt.plot(rewards_smooth, label=label, alpha=0.8)

    # plt.title("Training Reward Comparison: Memetic PPO vs Standard PPO")
    plt.xlabel("Episode")
    plt.ylabel("Mean Episode Reward")
    plt.xlim(0, 6000)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    # plt.show() # 如果在远程服务器运行，注释掉这一行


if __name__ == "__main__":
    from channel import set_global_seed

    base_cfg = Config()
    set_global_seed(base_cfg.seed)
    os.makedirs(base_cfg.out_dir, exist_ok=True)

    all_rewards = {}

    # 1. 运行 Standard PPO
    cfg_ppo = Config()
    cfg_ppo.use_memetic_search = False
    # 确保输出目录不冲突或整理好
    cfg_ppo.out_dir = "results_ppo_standard"
    os.makedirs(cfg_ppo.out_dir, exist_ok=True)

    rewards_ppo = run_experiment("ppo_standard", cfg_ppo)
    all_rewards["Standard PPO"] = rewards_ppo

    # 2. 运行 Memetic PPO
    # 注意：这里 memetic_eval_only=False 表示在训练过程中也应用 memetic 搜索
    # 这意味着 Policy 学习的是“如何给 Memetic 搜索提供一个好的初始动作”
    cfg_memetic = Config()
    cfg_memetic.use_memetic_search = True
    cfg_memetic.memetic_eval_only = False
    cfg_memetic.out_dir = "results_ppo_memetic"
    os.makedirs(cfg_memetic.out_dir, exist_ok=True)

    rewards_memetic = run_experiment("ppo_memetic", cfg_memetic)
    all_rewards["S-MPPO"] = rewards_memetic

    # 3. 绘图对比
    plot_results(all_rewards, "comparison_plot.svg")
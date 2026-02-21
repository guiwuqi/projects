import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env

from fas_environment import FASA2GEnv


def select_device(force_cpu=False):
    """自动选择训练设备，优先使用GPU。"""
    if force_cpu:
        print("[Device] 使用CPU训练（用户指定）")
        return "cpu"

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[Device] 使用GPU训练: {device_name}")
        return "cuda"

    print("[Device] 未检测到可用GPU，回退到CPU")
    return "cpu"


class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        self.modeling_errors = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.rewards.append(self.locals['rewards'][0])
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                if 'modeling_error' in self.locals['infos'][0]:
                    self.modeling_errors.append(self.locals['infos'][0]['modeling_error'])
        return True


def evaluate_agent(model, env, num_episodes=32):
    """评估智能体性能，返回平均容量、平均模型误差与最优配置。"""
    capacities = []
    modeling_errors = []
    best_config = None
    best_capacity = -np.inf

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_capacity = 0.0
        episode_error = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_capacity = float(info['channel_capacity'])
            episode_error = float(info['modeling_error'])
            done = terminated or truncated

            if episode_capacity > best_capacity:
                best_capacity = episode_capacity
                best_config = {k: np.copy(v) for k, v in env.state.items()}

        capacities.append(episode_capacity)
        modeling_errors.append(episode_error)

    return {
        'mean_capacity': float(np.mean(capacities)),
        'std_capacity': float(np.std(capacities)),
        'mean_error': float(np.mean(modeling_errors)),
        'std_error': float(np.std(modeling_errors)),
        'best_capacity': float(best_capacity),
        'best_config': best_config,
    }


def save_evaluation_results(result, method_name, log_dir="logs"):
    """保存评估结果到时间戳文件。"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_{method_name}_{timestamp}.txt"
    filepath = os.path.join(log_dir, filename)

    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"方法: {method_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"平均信道容量: {result['mean_capacity']:.4f} ± {result['std_capacity']:.4f} bps/Hz\n")
        f.write(f"平均模型误差: {result['mean_error']:.4f} ± {result['std_error']:.4f} dB\n")
        f.write(f"最佳信道容量: {result['best_capacity']:.4f} bps/Hz\n")
        f.write("最佳配置:\n")
        if result['best_config'] is not None:
            for key, value in result['best_config'].items():
                f.write(f"  {key}: {value[0]}\n")

    print(f"[{method_name}] 评估结果已保存: {filepath}")
    return filepath


def build_ppo_model(env, device, seed=42):
    policy_kwargs = dict(
        net_arch=[128, 128],
        activation_fn=nn.ReLU,
        ortho_init=False
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=1.0,
        policy_kwargs=policy_kwargs,
        device=device,
        seed=seed,
        verbose=1,
    )
    return model


def train_method(method_name, use_local_search, total_timesteps, device, seed=42):
    """训练一个方法（baseline PPO 或改进版 Memetic PPO）。"""
    env = FASA2GEnv(use_local_search=use_local_search, search_radius=1)
    check_env(env)

    model = build_ppo_model(env, device=device, seed=seed)

    save_dir = os.path.join("experiments", method_name)
    best_dir = os.path.join(save_dir, "best_model")
    logs_dir = os.path.join(save_dir, "logs")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    reward_callback = RewardLoggerCallback(check_freq=1024)
    eval_callback = EvalCallback(
        env,
        best_model_save_path=best_dir,
        log_path=logs_dir,
        eval_freq=2048,
        n_eval_episodes=16,
        deterministic=True,
        render=False,
    )

    print(f"\n===== 开始训练: {method_name} =====")
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, reward_callback])

    model_path = os.path.join(save_dir, f"{method_name}_final")
    model.save(model_path)
    print(f"[{method_name}] 模型已保存: {model_path}.zip")

    plot_training_results(
        reward_callback.rewards,
        reward_callback.modeling_errors,
        save_dir=logs_dir,
        prefix=method_name,
    )

    return model, env


def compare_methods(total_timesteps=20480, eval_episodes=32, force_cpu=False):
    """同时训练并对比 baseline PPO 与改进 Memetic PPO。"""
    device = select_device(force_cpu=force_cpu)

    baseline_model, baseline_env = train_method(
        method_name="ppo_baseline",
        use_local_search=False,
        total_timesteps=total_timesteps,
        device=device,
        seed=42,
    )
    improved_model, improved_env = train_method(
        method_name="memetic_ppo",
        use_local_search=True,
        total_timesteps=total_timesteps,
        device=device,
        seed=42,
    )

    baseline_result = evaluate_agent(baseline_model, baseline_env, num_episodes=eval_episodes)
    improved_result = evaluate_agent(improved_model, improved_env, num_episodes=eval_episodes)

    save_evaluation_results(baseline_result, "ppo_baseline")
    save_evaluation_results(improved_result, "memetic_ppo")

    print("\n================ 对比结果 ================")
    print(f"Baseline PPO 平均容量: {baseline_result['mean_capacity']:.4f} bps/Hz")
    print(f"Memetic PPO 平均容量:  {improved_result['mean_capacity']:.4f} bps/Hz")
    print(f"容量提升: {improved_result['mean_capacity'] - baseline_result['mean_capacity']:.4f} bps/Hz")

    print(f"Baseline PPO 平均误差: {baseline_result['mean_error']:.4f} dB")
    print(f"Memetic PPO 平均误差:  {improved_result['mean_error']:.4f} dB")
    print(f"误差变化: {improved_result['mean_error'] - baseline_result['mean_error']:.4f} dB")

    save_comparison_table(baseline_result, improved_result, out_path="experiments/comparison_summary.txt")



def save_comparison_table(baseline_result, improved_result, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("FAS端口优化: Baseline PPO vs Memetic PPO\n")
        f.write("=" * 60 + "\n")
        f.write(f"Baseline mean_capacity: {baseline_result['mean_capacity']:.4f}\n")
        f.write(f"Improved mean_capacity: {improved_result['mean_capacity']:.4f}\n")
        f.write(f"Capacity gain: {improved_result['mean_capacity'] - baseline_result['mean_capacity']:.4f}\n")
        f.write(f"Baseline mean_error: {baseline_result['mean_error']:.4f}\n")
        f.write(f"Improved mean_error: {improved_result['mean_error']:.4f}\n")
        f.write(f"Error diff: {improved_result['mean_error'] - baseline_result['mean_error']:.4f}\n")


def plot_training_results(rewards, modeling_errors, save_dir='./logs', prefix='method'):
    """保存训练曲线原始数据，方便后续论文绘图。"""
    os.makedirs(save_dir, exist_ok=True)

    if rewards:
        rewards_path = os.path.join(save_dir, f'{prefix}_reward_progression.csv')
        with open(rewards_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('step,reward\n')
            for idx, value in enumerate(rewards):
                f.write(f'{idx},{float(value)}\n')

    if modeling_errors:
        error_path = os.path.join(save_dir, f'{prefix}_modeling_error_progression.csv')
        with open(error_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('step,modeling_error\n')
            for idx, value in enumerate(modeling_errors):
                f.write(f'{idx},{float(value)}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FAS A2G 信道优化: PPO对比实验')
    parser.add_argument('--mode', type=str, default='compare', choices=['compare', 'train', 'evaluate'],
                        help='compare: 基线与改进版对比训练；train/evaluate: 兼容旧入口（内部等同compare）')
    parser.add_argument('--timesteps', type=int, default=20480, help='每个方法的训练步数')
    parser.add_argument('--eval_episodes', type=int, default=32, help='最终评估回合数')
    parser.add_argument('--force_cpu', action='store_true', help='强制使用CPU（默认优先GPU）')
    args = parser.parse_args()

    compare_methods(
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        force_cpu=args.force_cpu,
    )
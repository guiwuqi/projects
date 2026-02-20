# train_main.py
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import torch.nn
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.callbacks import BaseCallback
# 从环境文件导入
from fas_environment import FASA2GEnv


class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        self.modeling_errors = []  # 新增：记录模型误差

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.rewards.append(self.locals['rewards'][0])
            # 尝试获取模型误差
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                if 'modeling_error' in self.locals['infos'][0]:
                    self.modeling_errors.append(self.locals['infos'][0]['modeling_error'])
        return True


def evaluate_agent(model, env, num_episodes=32):
    """评估智能体性能，现在也记录模型误差"""
    capacities = []
    modeling_errors = []  # 新增：记录模型误差
    best_config = None
    best_capacity = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_capacity = 0
        episode_error = 0  # 新增

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_capacity = info['channel_capacity']
            episode_error = info['modeling_error']  # 新增
            done = terminated or truncated

            if episode_capacity > best_capacity:
                best_capacity = episode_capacity
                best_config = env.state.copy()

        capacities.append(episode_capacity)
        modeling_errors.append(episode_error)  # 新增

    return np.mean(capacities), np.mean(modeling_errors), best_capacity, best_config


def save_evaluation_results(mean_capacity, mean_error, best_capacity, best_config, log_dir="logs"):
    """保存评估结果到log文件夹下的时间戳文件"""

    # 创建log文件夹（如果不存在）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{timestamp}.txt"
    filepath = os.path.join(log_dir, filename)

    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write("FAS A2G 信道优化评估结果\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"平均信道容量: {mean_capacity:.2f} bps/Hz\n")
        f.write(f"平均模型误差: {mean_error:.2f} dB\n")  # 新增
        f.write(f"最佳信道容量: {best_capacity:.2f} bps/Hz\n\n")

        f.write("最佳配置参数:\n")
        f.write("-" * 30 + "\n")
        for key, value in best_config.items():
            f.write(f"{key}: {value[0]}\n")

    print(f"评估结果已保存到 {filepath}")
    return filepath


def train_rl_agent():
    """训练强化学习智能体"""
    # 创建环境
    env = FASA2GEnv()
    # 检查环境
    check_env(env)
    # 创建PPO智能体 - 修改超参数
    policy_kwargs = dict(
        net_arch=[128, 128],  # 减小网络规模
        activation_fn=torch.nn.ReLU,
        ortho_init=False  # 关闭正交初始化
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=1e-4,  # 稍微提高学习率
        n_steps=1024,  # 减少步数
        batch_size=128,  # 增加批量大小
        n_epochs=3,  # 减少epoch数
        gamma=0.99,  # 降低折扣因子
        gae_lambda=0.95,  # 调整GAE参数
        clip_range=0.2,  # 减小裁剪范围
        ent_coef=0.05,  # 增加熵系数鼓励探索
        vf_coef=0.5,  # 调整价值函数系数
        max_grad_norm=1.0,  # 梯度裁剪
        policy_kwargs=policy_kwargs,
        verbose=1
    )


    # 添加奖励日志回调
    reward_callback = RewardLoggerCallback(check_freq=1024)
    # 修改评估回调
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./best_model/',
        log_path='./logs/',
        eval_freq=2048,  # 更频繁的评估
        n_eval_episodes=32,  # 每次评估多个episode
        deterministic=True,
        render=False
    )

    # 训练时同时使用2个回调
    callback_list = [eval_callback, reward_callback]

    print("开始训练...")
    model.learn(total_timesteps=20480, callback=callback_list)#81920
    model.save("fas_a2g_best_optimizer")

    # 绘制训练结果图
    plot_training_results(reward_callback.rewards, reward_callback.modeling_errors)

    return model, env


def load_and_evaluate_best_model():
    """加载最佳模型并进行评估"""
    # 创建环境
    env = FASA2GEnv()

    # 加载最佳模型
    best_model_path = "./best_model/best_model"
    if not os.path.exists(best_model_path + ".zip"):
        print(f"最佳模型文件不存在: {best_model_path}.zip")
        # 尝试加载默认保存的模型
        best_model_path = "./data/fas_a2g_best_optimizer.zip"

    try:
        model = PPO.load(best_model_path, env=env)
        print("成功加载最佳模型")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 评估性能
    print("开始评估模型...")
    mean_capacity, mean_error, best_capacity, best_config = evaluate_agent(model, env, num_episodes=32)

    print(f"平均信道容量: {mean_capacity:.2f} bps/Hz")
    print(f"平均模型误差: {mean_error:.2f} dB")
    print(f"最佳信道容量: {best_capacity:.2f} bps/Hz")
    print("最佳配置:")
    for key, value in best_config.items():
        print(f"  {key}: {value[0]}")

    # 保存结果
    save_evaluation_results(mean_capacity, mean_error, best_capacity, best_config)


def plot_training_results(rewards, modeling_errors, save_dir='./logs/'):
    """绘制训练结果图 - 分开显示各个指标"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 绘制奖励变化图
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, linewidth=2, alpha=0.7, label='Reward', color='blue')
    if len(rewards) > 10:
        window = max(1, len(rewards) // 20)
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(rewards)), moving_avg, 'r-', linewidth=2,
                 label=f'Moving Avg (window={window})')
    plt.title('Reward Progression During Training', fontsize=14)
    plt.xlabel('Training Steps (x1000)', fontsize=12)
    plt.ylabel('Reward Value', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reward_progression.svg'), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 绘制模型误差变化图
    if modeling_errors:
        plt.figure(figsize=(10, 6))
        plt.plot(modeling_errors, linewidth=2, alpha=0.7, color='orange', label='Modeling Error')
        plt.title('Modeling Error Progression During Training', fontsize=14)
        plt.xlabel('Training Steps (x1000)', fontsize=12)
        plt.ylabel('Modeling Error (dB)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'modeling_error_progression.svg'), dpi=300, bbox_inches='tight')
        plt.show()

    # 3. 绘制奖励和误差的联合图（可选）
    if modeling_errors and len(rewards) == len(modeling_errors):
        plt.figure(figsize=(10, 6))
        # 标准化数据以便在同一图中显示
        rewards_norm = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
        errors_norm = (modeling_errors - np.min(modeling_errors)) / (np.max(modeling_errors) - np.min(modeling_errors))

        plt.plot(rewards_norm, linewidth=2, alpha=0.7, label='Reward (Normalized)')
        plt.plot(errors_norm, linewidth=2, alpha=0.7, label='Modeling Error (Normalized)', color='orange')
        plt.title('Normalized Reward vs Modeling Error', fontsize=14)
        plt.xlabel('Training Steps (x1000)', fontsize=12)
        plt.ylabel('Normalized Value', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'normalized_comparison.svg'), dpi=300, bbox_inches='tight')
        plt.show()

    print(f"训练结果图已保存至: {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FAS A2G 信道优化')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='运行模式: train (训练) 或 evaluate (评估)')

    args = parser.parse_args()

    if args.mode == 'train':
        # 训练和评估
        model, env = train_rl_agent()

        # 评估性能
        mean_capacity, mean_error, best_capacity, best_config = evaluate_agent(model, env)

        print(f"平均信道容量: {mean_capacity:.2f} bps/Hz")
        print(f"平均模型误差: {mean_error:.2f} dB")
        print(f"最佳信道容量: {best_capacity:.2f} bps/Hz")
        print("最佳配置:")
        for key, value in best_config.items():
            print(f"  {key}: {value[0]}")

        save_evaluation_results(mean_capacity, mean_error, best_capacity, best_config)

    elif args.mode == 'evaluate':
        load_and_evaluate_best_model()

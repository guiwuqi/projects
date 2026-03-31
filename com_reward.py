import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 可选，用于显示进度条

from config import Config
from channel import (generate_uav_trajectory, sample_paths, build_tx_array_offsets,
                     compute_capacity, build_rx_port_positions)
from env import FASEnv  # 只是为了使用相同的环境逻辑，但奖励计算我们直接手动实现


def moving_average(x, window=20):
    """计算移动平均"""
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode='valid')


def evaluate_baseline(cfg, strategy, n_episodes=100, random_k=10):
    """
    评估基线策略，返回每个 episode 的累计奖励列表。
    strategy: "random_k" 或 "full"
    """
    # 预计算固定部分（无人机轨迹和接收端口位置）
    times, uav_positions = generate_uav_trajectory(cfg)
    rx_port_positions = build_rx_port_positions(cfg)
    tx_offsets = build_tx_array_offsets(cfg)

    returns = []
    rng = np.random.default_rng(cfg.seed)  # 固定随机种子以保证可重复性

    for ep in tqdm(range(n_episodes), desc=f"Evaluating {strategy}"):
        # 每个 episode 重新采样多径簇（与训练一致）
        clusters = sample_paths(cfg, rng)
        total_reward = 0.0

        for t_idx in range(cfg.T):
            # 计算当前时间步的信道矩阵
            uav_pos_t = uav_positions[t_idx]
            tx_positions = uav_pos_t[None, :] + tx_offsets  # (M,3)

            k = 2.0 * np.pi / cfg.wavelength
            H_t = np.zeros((cfg.N, cfg.M), dtype=np.complex128)

            for cl in clusters:
                alphas = cl["alphas"]
                gains = cl["gains"]
                directions = np.stack([np.cos(alphas), np.sin(alphas), np.zeros_like(alphas)], axis=1)
                rx_phase = np.exp(-1j * k * (rx_port_positions @ directions.T))
                tx_phase = np.exp(-1j * k * (tx_positions @ directions.T))
                weighted_tx = gains[:, None] * tx_phase.T
                H_t += rx_phase @ weighted_tx

            # 根据策略选择激活端口集
            if strategy == "full":
                active_set = set(range(cfg.N))
            elif strategy == "random_k":
                # 每步随机选择 random_k 个不重复的端口
                active_set = set(rng.choice(cfg.N, size=random_k, replace=False))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # 计算容量
            if len(active_set) == 0:
                capacity = 0.0
            else:
                idx = np.array(sorted(active_set), dtype=int)
                H_active = H_t[idx, :]
                capacity = compute_capacity(H_active, cfg.rho_linear, cfg.M)

            # 奖励函数（与训练时一致，但去掉额外奖励部分，仅使用 capacity / len(active_set)）
            # 注意：训练时有时会加入容量增加的额外奖励，这里为了公平比较，只使用基础部分
            reward = capacity / len(active_set)
            total_reward += reward

        returns.append(total_reward)

    return np.array(returns)


def plot_comparison(base_dir="results_fas_memetic_ppo", n_episodes_baseline=100, random_k=10):
    # 加载训练奖励文件
    file_memetic = os.path.join(base_dir, "train_rewards.npy")
    file_none = os.path.join(base_dir, "train_rewards_none.npy")

    if not os.path.exists(file_memetic) or not os.path.exists(file_none):
        print("奖励文件缺失，请确保已分别运行过启用和不启用局部搜索的训练。")
        return

    rewards_memetic = np.load(file_memetic)
    rewards_none = np.load(file_none)

    # 创建配置实例
    cfg = Config()
    # 确保输出目录与训练时一致
    cfg.out_dir = base_dir

    # 评估基线策略
    print("Evaluating random-k baseline...")
    rewards_random = evaluate_baseline(cfg, "random_k", n_episodes=n_episodes_baseline, random_k=random_k)
    print("Evaluating full activation baseline...")
    rewards_full = evaluate_baseline(cfg, "full", n_episodes=n_episodes_baseline)

    # 平滑所有曲线
    window = 20
    ma_none = moving_average(rewards_none, window)
    ma_memetic = moving_average(rewards_memetic, window)
    ma_random = moving_average(rewards_random, window)
    ma_full = moving_average(rewards_full, window)

    # 创建横坐标（移动平均后长度变短）
    x_none = np.arange(len(ma_none)) + (window - 1)
    x_memetic = np.arange(len(ma_memetic)) + (window - 1)
    x_random = np.arange(len(ma_random)) + (window - 1)
    x_full = np.arange(len(ma_full)) + (window - 1)

    # 绘图
    plt.figure(figsize=(10, 6))

    # 原始奖励（半透明）
    plt.plot(rewards_none, alpha=0.2, color='blue')
    plt.plot(rewards_memetic, alpha=0.2, color='red')
    plt.plot(rewards_random, alpha=0.2, color='green')
    plt.plot(rewards_full, alpha=0.2, color='orange')

    # 平滑曲线
    plt.plot(x_none, ma_none, lw=2, label="w/o memetic (smooth)", color='blue')
    plt.plot(x_memetic, ma_memetic, lw=2, label="w/ memetic (smooth)", color='red')
    plt.plot(x_random, ma_random, lw=2, label=f"Random {random_k} ports (smooth)", color='green')
    plt.plot(x_full, ma_full, lw=2, label="Full activation (smooth)", color='orange')

    # 可选：显示随机基线的置信区间（标准差）
    # 计算随机基线每个 episode 的奖励的均值和标准差，然后平滑阴影区域
    # 这里简单起见，只画平滑后的平均线

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    #plt.title("Training Reward Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "reward_comparison.svg"), dpi=200)
    plt.close()
    print("对比图已保存至 figures/reward_comparison.svg")


if __name__ == "__main__":
    plot_comparison()
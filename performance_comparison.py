import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import Config
from channel import (generate_uav_trajectory, sample_paths,
                     compute_capacity, build_rx_port_positions, build_tx_array_offsets)


def moving_average(x, window=50):
    """计算移动平均以平滑曲线"""
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode='valid')


def evaluate_static_strategies(cfg, n_episodes=500, random_k=10):
    """
    评估非学习策略：随机选择K个端口 和 全激活。
    为了公平对比，计算方式与环境中的 reward = capacity / num_active 一致。
    """
    times, uav_positions = generate_uav_trajectory(cfg)
    rx_port_positions = build_rx_port_positions(cfg)
    tx_offsets = build_tx_array_offsets(cfg)
    k_vec = 2.0 * np.pi / cfg.wavelength

    rewards_random = []
    rewards_full = []

    print(f"Simulating Baselines (Random-{random_k} and Full)...")
    rng = np.random.default_rng(cfg.seed + 777)

    for ep in tqdm(range(n_episodes)):
        # 每个 episode 采样一套新的多径路径（与环境逻辑一致）
        clusters = sample_paths(cfg, rng)

        ep_ret_random = 0
        ep_ret_full = 0

        for t_idx in range(cfg.T):
            # 1. 计算当前时刻的全信道矩阵 H
            uav_pos_t = uav_positions[t_idx]
            tx_positions = uav_pos_t[None, :] + tx_offsets
            H_t = np.zeros((cfg.N, cfg.M), dtype=np.complex128)
            for cl in clusters:
                alphas = cl["alphas"]
                gains = cl["gains"]
                directions = np.stack([np.cos(alphas), np.sin(alphas), np.zeros_like(alphas)], axis=1)
                rx_phase = np.exp(-1j * k_vec * (rx_port_positions @ directions.T))
                tx_phase = np.exp(-1j * k_vec * (tx_positions @ directions.T))
                H_t += rx_phase @ (gains[:, None] * tx_phase.T)

            # 2. 策略 A: 随机选择 10 个端口
            idx_rand = rng.choice(cfg.N, size=random_k, replace=False)
            cap_rand = compute_capacity(H_t[idx_rand, :], cfg.rho_linear, cfg.M)
            ep_ret_random += cap_rand / random_k

            # 3. 策略 B: 全激活 (N个端口)
            cap_full = compute_capacity(H_t, cfg.rho_linear, cfg.M)
            ep_ret_full += cap_full / cfg.N

        rewards_random.append(ep_ret_random)
        rewards_full.append(ep_ret_full)

    return np.array(rewards_random), np.array(rewards_full)


def plot_all_comparison(base_dir="results_fas_memetic_ppo"):
    # 1. 加载训练好的 RL 数据
    file_memetic = os.path.join(base_dir, "train_rewards.npy")
    file_ppo_base = os.path.join(base_dir, "train_rewards_none.npy")

    if not os.path.exists(file_memetic) or not os.path.exists(file_ppo_base):
        print(f"Error: RL reward files not found in {base_dir}")
        return

    res_memetic = np.load(file_memetic)
    res_ppo_base = np.load(file_ppo_base)

    # 2. 模拟基线数据
    cfg = Config()
    # 模拟的 episode 数量建议与 RL 的量级匹配，或者取平均值拉成直线
    # 这里我们模拟 500 个 episode 取平均表现
    n_base_ep = 500
    raw_random, raw_full = evaluate_static_strategies(cfg, n_episodes=n_base_ep, random_k=10)

    # 计算基线的平均值，用于在图中画出稳定的水平参考线
    mean_random = np.mean(raw_random)
    mean_full = np.mean(raw_full)

    # 3. 平滑处理 RL 曲线
    win = 50
    ma_memetic = moving_average(res_memetic, win)
    ma_ppo_base = moving_average(res_ppo_base, win)

    # 4. 绘图
    plt.figure(figsize=(10, 6))

    # 绘制两条学习曲线
    epochs_m = np.arange(len(ma_memetic)) + win
    epochs_p = np.arange(len(ma_ppo_base)) + win

    plt.plot(epochs_m, ma_memetic, label="Memetic PPO (Local Search)", color='firebrick', lw=2)
    plt.plot(epochs_p, ma_ppo_base, label="Baseline PPO (No Local Search)", color='royalblue', lw=2)

    # 绘制两条基线 (由于基线不学习，表现是平稳的，绘制成虚线横线更清晰)
    # 如果你想看基线的波动，也可以直接 plot(raw_random)
    total_steps = max(len(res_memetic), len(res_ppo_base))
    plt.hlines(mean_random, 0, total_steps, colors='green', linestyles='--', label="Baseline: Random 10 Ports")
    plt.hlines(mean_full, 0, total_steps, colors='darkorange', linestyles='-.', label="Baseline: Full Activation")

    # 装饰
    plt.xlabel("Training Episode")
    plt.ylabel("Cumulative Reward (Capacity / #Ports)")
    plt.title("Reward Comparison: Memetic vs Baselines")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # 保存
    out_path = os.path.join(base_dir, "figures", "com_reward_all.svg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Success: Comparison plot saved to {out_path}")


if __name__ == "__main__":
    # 执行绘图
    plot_all_comparison()
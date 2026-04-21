import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from config import Config
from env import FASEnv
from channel import (
    set_global_seed,
    build_tx_array_offsets,
    build_path_directions,
    compute_capacity,
    doppler_phase,
)


def get_ula_capacity(cfg, n_0, clusters):
    """
    Calculate the channel capacity of ULA over 0-3s with specified number of antennas n_0.
    ULA elements are uniformly distributed along [0, W].
    """
    # 1. Construct ULA receiver element positions (uniform spacing)
    xs = np.linspace(0, cfg.W, n_0)
    rx_pos = np.zeros((n_0, 3))
    rx_pos[:, 0] = xs

    # 2. 计算ULA接收端的参考点（中心点）
    rx_ref = np.mean(rx_pos, axis=0)  # ULA中心

    k = 2.0 * np.pi / cfg.wavelength
    tx_offsets = build_tx_array_offsets(cfg)

    ula_capacities = []

    # 3. Compute channel matrices for 30 time steps
    for t in range(cfg.T):
        H_t = np.zeros((n_0, cfg.M), dtype=np.complex128)
        time_s = t * cfg.dt

        for cl in clusters:
            alphas = cl["alphas"]
            betas = cl["betas"]
            gains = cl["gains"]
            distance_phase = cl["distance_phase"]  # (P,) 远场近似：时间不变路径相位

            # 方向向量
            directions = build_path_directions(alphas, betas)  # (P, 3)
            # 接收端几何相位
            rx_offsets = rx_pos - rx_ref[None, :]  # (n_0, 3)
            rx_phase = np.exp(-1j * k * (rx_offsets @ directions.T))  # (n_0, P)

            # 发射端几何相位
            tx_phase = np.exp(-1j * k * (tx_offsets @ directions.T))  # (M, P)

            # 多普勒相位
            doppler = doppler_phase(cfg, directions, time_s, k)  # (P,)

            # 合并所有相位项（包含距离相位！）
            weighted_tx = (gains * doppler * distance_phase)[:, None] * tx_phase.T
            H_t += rx_phase @ weighted_tx

        cap = compute_capacity(H_t, cfg.rho_linear, cfg.M)
        ula_capacities.append(cap)

    return np.array(ula_capacities)


def run_comparison():
    cfg = Config()
    set_global_seed(cfg.seed)

    # --- 1. Load FAS (Memetic PPO) model results ---
    print("Evaluating FAS (Memetic PPO)...")
    model_path = os.path.join("results_ppo_memetic", "models", "ppo_ppo_memetic")
    model = PPO.load(model_path)

    env = FASEnv(cfg)
    obs, info = env.reset(seed=cfg.seed)

    # FAS 初始端口数（仅用于回退）
    n_0 = info["active_count"]
    ula_port_count = cfg.ula_fixed_port_count if cfg.ula_fixed_port_count is not None else n_0
    if ula_port_count < 1:
        raise ValueError("ula_fixed_port_count 必须 >= 1")

    # Use consistent channel parameters
    clusters = env.clusters
    fas_capacities = [info["capacity"]]
    fas_active_counts = [info["active_count"]]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        fas_capacities.append(info["capacity"])
        fas_active_counts.append(info["active_count"])

    fas_caps = np.array(fas_capacities)
    times = np.arange(len(fas_caps)) * cfg.dt

    # --- 2. Compute ULA baseline (fixed user-specified port count) ---
    print(f"Calculating ULA baseline (Antennas = {ula_port_count})...")
    ula_caps = get_ula_capacity(cfg, ula_port_count, clusters)

    # 确保长度一致
    if len(ula_caps) < len(fas_caps):
        ula_caps = np.append(ula_caps, ula_caps[-1])

    # 计算变化量
    if len(ula_caps) > 1:
        ula_variation = np.std(ula_caps) / np.mean(ula_caps)
        print(f"ULA capacity variation (std/mean): {ula_variation:.4f}")

    # --- 4. Plotting ---
    os.makedirs("comparison_plots", exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # Figure 1: Capacity over time (0-3s)
    plt.figure(figsize=(10, 5))
    plt.plot(times, fas_caps, 'r-o', label=f'FAS (Memetic PPO, Q={cfg.K})', markersize=4)
    plt.plot(times, ula_caps, 'b--s', label=f'ULA (Fixed Antennas = {ula_port_count})', markersize=4)
    plt.xlabel('Time (s)')
    plt.ylabel('Channel Capacity (bps/Hz)')
    #plt.title('FAS vs ULA Capacity Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_plots/capacity_vs_time.svg", dpi=300)

    print("\n--- Summary Results ---")
    print(f"ULA Average Capacity: {ula_caps.mean():.4f} bps/Hz")
    print(f"FAS Average Capacity: {fas_caps.mean():.4f} bps/Hz")
    print(f"Performance Gain: {(fas_caps.mean() - ula_caps.mean()) / ula_caps.mean() * 100:.2f}%")


if __name__ == "__main__":
    run_comparison()
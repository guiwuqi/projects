
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from config import Config
from env import FASEnv
from channel import set_global_seed, build_tx_array_offsets, compute_capacity, sample_paths


def get_ula_capacity(cfg, n_0, clusters, uav_positions):
    """
    Calculate the channel capacity of ULA over 0-3s with specified number of antennas n_0.
    ULA elements are uniformly distributed along [0, W].
    """
    # 1. Construct ULA receiver element positions (uniform spacing)
    xs = np.linspace(0, cfg.W, n_0)
    rx_pos = np.zeros((n_0, 3))
    rx_pos[:, 0] = xs

    k = 2.0 * np.pi / cfg.wavelength
    tx_offsets = build_tx_array_offsets(cfg)

    ula_capacities = []

    # 2. Compute channel matrices for 30 time steps
    for t in range(cfg.T):
        tx_pos_t = uav_positions[t][None, :] + tx_offsets
        H_t = np.zeros((n_0, cfg.M), dtype=np.complex128)

        for cl in clusters:
            alphas = cl["alphas"]
            gains = cl["gains"]
            directions = np.stack([np.cos(alphas), np.sin(alphas), np.zeros_like(alphas)], axis=1)

            rx_phase = np.exp(-1j * k * (rx_pos @ directions.T))
            tx_phase = np.exp(-1j * k * (tx_pos_t @ directions.T))
            weighted_tx = gains[:, None] * tx_phase.T
            H_t += rx_phase @ weighted_tx

        cap = compute_capacity(H_t, cfg.rho_linear, cfg.M)
        ula_capacities.append(cap)

    return np.array(ula_capacities)


def run_comparison():
    cfg = Config()
    set_global_seed(cfg.seed)

    # --- 1. Load FAS (Memetic PPO) model results ---
    print("Evaluating FAS (Memetic PPO)...")
    model_path = os.path.join(cfg.out_dir, "models", "ppo_fas_memetic")
    model = PPO.load(model_path)

    env = FASEnv(cfg)
    obs, info = env.reset(seed=cfg.seed)

    # Get initial number of active ports
    n_0 = info["active_count"]
    # Use consistent channel parameters
    clusters = env.clusters
    uav_positions = env.uav_positions

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

    # --- 2. Compute ULA baseline (using same n_0) ---
    print(f"Calculating ULA baseline (Antennas = {n_0})...")
    ula_caps = get_ula_capacity(cfg, n_0, clusters, uav_positions)
    # Align ULA with FAS if FAS has one extra time step
    if len(ula_caps) < len(fas_caps):
        ula_caps = np.append(ula_caps, ula_caps[-1])

    # --- 3. Average capacity comparison across different port counts ---
    print("Comparing average capacity across different port counts...")
    port_counts = [2, 4, 8, 12, 16, 20]
    ula_avg_results = []
    fas_avg_results = []

    for k_val in port_counts:
        # Compute ULA average capacity
        ula_c = get_ula_capacity(cfg, k_val, clusters, uav_positions).mean()
        ula_avg_results.append(ula_c)

        # Evaluate FAS with fixed number of active ports
        cfg_temp = Config()
        cfg_temp.init_active_ports = tuple(np.linspace(0, cfg.N - 1, k_val, dtype=int))
        cfg_temp.Kmin = k_val
        cfg_temp.Kmax = k_val

        env_temp = FASEnv(cfg_temp)
        obs_t, info_t = env_temp.reset(seed=cfg.seed)
        env_temp.clusters = clusters  # Enforce identical channel conditions

        temp_caps = []
        d_t = False
        while not d_t:
            act_t, _ = model.predict(obs_t, deterministic=True)
            obs_t, _, term_t, trunc_t, info_t = env_temp.step(int(act_t))
            temp_caps.append(info_t["capacity"])
            d_t = term_t or trunc_t
        fas_avg_results.append(np.mean(temp_caps))

    # --- 4. Plotting: Only Figure 1 (Time-domain capacity) remains ---
    os.makedirs("comparison_plots", exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['Arial']  # Use English-friendly font
    plt.rcParams['axes.unicode_minus'] = False

    # Figure 1: Capacity over time (0-3s)
    plt.figure(figsize=(10, 5))
    plt.plot(times, fas_caps, 'r-o', label='FAS (Memetic PPO, Dynamic Ports)')
    plt.plot(times, ula_caps, 'b--s', label=f'ULA (Fixed Antennas = {n_0})')
    plt.xlabel('Time (s)')
    plt.ylabel('Channel Capacity (bps/Hz)')
    #plt.title('FAS vs ULA Capacity Over 0-3 Seconds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    #plt.savefig("comparison_plots/capacity_vs_time.png", dpi=300)

    print("\n--- Summary Results ---")
    print(f"ULA Average Capacity: {ula_caps.mean():.4f} bps/Hz")
    print(f"FAS Average Capacity: {fas_caps.mean():.4f} bps/Hz")
    print(f"Performance Gain: {(fas_caps.mean() - ula_caps.mean()) / ula_caps.mean() * 100:.2f}%")
    plt.show()


if __name__ == "__main__":
    run_comparison()

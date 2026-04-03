import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from channel import set_global_seed
from config import Config
from env import FASEnv


def moving_average(x: List[float], window: int = 20) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def evaluate_agent(cfg: Config, model: PPO, deterministic: bool = True) -> Dict[str, np.ndarray]:
    env = FASEnv(cfg)
    obs, info = env.reset(seed=cfg.seed)

    capacities = [info["capacity"]]
    active_counts = [info["active_count"]]
    active_ports = [info["active_ports"]]   # 新增：记录端口索引列表
    actions = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        actions.append(info["chosen_action"])
        capacities.append(info["capacity"])
        active_counts.append(info["active_count"])
        active_ports.append(info["active_ports"])   # 新增

    times = np.arange(len(capacities)) * cfg.dt
    return {
        "times": times,
        "capacities": np.array(capacities),
        "active_counts": np.array(active_counts),
        "active_ports": active_ports,          # 列表的列表
        "actions": np.array(actions),
    }


def plot_results(cfg: Config, train_rewards: List[float], eval_data: Dict[str, np.ndarray]) -> None:
    fig_dir = os.path.join(cfg.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(eval_data["times"], eval_data["capacities"], marker="o", lw=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Capacity (bps/Hz)")
    plt.title("Capacity vs Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "capacity_vs_time.svg"), dpi=200)
    plt.close()


    ma = moving_average(train_rewards, window=20)
    plt.figure(figsize=(8, 4.5))
    plt.plot(train_rewards, alpha=0.35, label="Episode reward")
    if len(ma) > 0:
        plt.plot(np.arange(len(ma)) + 19, ma, lw=2.0, label="Moving average (20)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "training_reward_curve.svg"), dpi=200)
    plt.close()


def run_evaluation(cfg: Config) -> Dict[str, np.ndarray]:
    model_path = os.path.join(cfg.out_dir, "models", "ppo_fas_memetic")
    model = PPO.load(model_path)

    eval_data = evaluate_agent(cfg, model, deterministic=True)

    train_rewards_path = os.path.join(cfg.out_dir, "train_rewards.npy")
    if os.path.exists(train_rewards_path):
        train_rewards = np.load(train_rewards_path).tolist()
    else:
        train_rewards = []

    plot_results(cfg, train_rewards, eval_data)

    np.savez(
        os.path.join(cfg.out_dir, "evaluation_data.npz"),
        times=eval_data["times"],
        capacities=eval_data["capacities"],
        active_counts=eval_data["active_counts"],
        actions=eval_data["actions"],
        train_rewards=np.array(train_rewards, dtype=float),
    )

    return eval_data


if __name__ == "__main__":
    cfg = Config()
    set_global_seed(cfg.seed)
    eval_data = run_evaluation(cfg)
    print(f"Avg capacity: {eval_data['capacities'].mean():.4f} bps/Hz")
    for t, ports in enumerate(eval_data['active_ports']):
        print(f"Time step {t}: {ports}")
import argparse
import os
from dataclasses import asdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from channel import set_global_seed
from config import Config
from env import FASEnv
from figure import get_ula_capacity


def _build_cfg(**overrides) -> Config:
    cfg = Config(**asdict(Config()))
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def evaluate_fas_avg_capacity(cfg: Config, model: PPO) -> float:
    env = FASEnv(cfg)
    obs, info = env.reset(seed=cfg.seed)

    capacities = [info["capacity"]]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        capacities.append(info["capacity"])

    return float(np.mean(capacities))


def evaluate_ula_avg_capacity(cfg: Config) -> float:
    env = FASEnv(cfg)
    _, _ = env.reset(seed=cfg.seed)
    ula_caps = get_ula_capacity(cfg, cfg.ula_fixed_port_count, env.clusters)
    return float(np.mean(ula_caps))


def plot_figure_two(
    snr_min: float,
    snr_max: float,
    snr_step: float,
    smppo_model: str,      # 新参数：S-MPPO 模型路径
    no_smppo_model: str,    # 新参数：no S-MPPO 模型路径
    out_path: str,
) -> None:
    if snr_step <= 0:
        raise ValueError("snr_step must be positive.")

    base_cfg = Config()
    set_global_seed(base_cfg.seed)

    # 加载两个独立的模型
    if not os.path.exists(smppo_model + ".zip"):
        raise FileNotFoundError(f"S-MPPO model not found: {smppo_model}.zip")
    if not os.path.exists(no_smppo_model + ".zip"):
        raise FileNotFoundError(f"no S-MPPO model not found: {no_smppo_model}.zip")

    model_smppo = PPO.load(smppo_model)
    model_no_smppo = PPO.load(no_smppo_model)

    snr_values = np.arange(snr_min, snr_max + 1e-9, snr_step)

    ula_curve: List[float] = []
    fas_memetic_curve: List[float] = []
    fas_no_memetic_curve: List[float] = []

    for snr_db in snr_values:
        # ULA
        cfg_ula = _build_cfg(**asdict(base_cfg))
        cfg_ula.snr_db = float(snr_db)
        ula_avg = evaluate_ula_avg_capacity(cfg_ula)
        ula_curve.append(ula_avg)

        # FAS with S-MPPO (use smppo model and memetic_eval_only=False)
        cfg_fas_memetic = _build_cfg(**asdict(base_cfg))
        cfg_fas_memetic.snr_db = float(snr_db)
        cfg_fas_memetic.use_memetic_search = True
        cfg_fas_memetic.memetic_eval_only = False   # 与训练时一致
        fas_memetic_avg = evaluate_fas_avg_capacity(cfg_fas_memetic, model_smppo)
        fas_memetic_curve.append(fas_memetic_avg)

        # FAS without S-MPPO (use no_smppo model and memetic_eval_only=True)
        cfg_fas_no_memetic = _build_cfg(**asdict(base_cfg))
        cfg_fas_no_memetic.snr_db = float(snr_db)
        cfg_fas_no_memetic.use_memetic_search = False
        cfg_fas_no_memetic.memetic_eval_only = True  # 与训练时一致
        fas_no_memetic_avg = evaluate_fas_avg_capacity(cfg_fas_no_memetic, model_no_smppo)
        fas_no_memetic_curve.append(fas_no_memetic_avg)

        print(
            f"snr_db={snr_db:>5.2f}  ULA={ula_avg:.4f}  "
            f"FAS(S-MPPO)={fas_memetic_avg:.4f}  FAS(no S-MPPO)={fas_no_memetic_avg:.4f}"
        )

    plt.figure(figsize=(9, 5.2))
    plt.plot(snr_values, ula_curve, marker="s", linewidth=1.8, linestyle="--", label="ULA")
    plt.plot(snr_values, fas_memetic_curve, marker="o", linewidth=1.8, label="FAS (S-MPPO)")
    plt.plot(snr_values, fas_no_memetic_curve, marker="^", linewidth=1.8, label="FAS (no S-MPPO)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average channel capacity (bps/Hz)")
    plt.xlim(-6, 10)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"Saved figure: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 2: average capacity vs SNR.")
    parser.add_argument("--snr-min", type=float, default=-6.0, help="Minimum SNR in dB.")
    parser.add_argument("--snr-max", type=float, default=10.0, help="Maximum SNR in dB.")
    parser.add_argument("--snr-step", type=float, default=1.0, help="SNR step in dB.")
    parser.add_argument(
        "--smppo-model",
        type=str,
        default=os.path.join("results_ppo_memetic", "models", "ppo_ppo_memetic"),
        help="Path to S-MPPO model (without .zip suffix).",
    )
    parser.add_argument(
        "--no-smppo-model",
        type=str,
        default=os.path.join("results_ppo_standard", "models", "ppo_ppo_standard"),
        help="Path to no S-MPPO model (without .zip suffix).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="comparison_plots/figure2_capacity_vs_snr.svg",
        help="Output figure path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_figure_two(
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        snr_step=args.snr_step,
        smppo_model=args.smppo_model,
        no_smppo_model=args.no_smppo_model,
        out_path=args.out,
    )
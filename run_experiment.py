import os

import numpy as np

from channel import set_global_seed
from config import Config
from eval import run_evaluation
from train import train_agent


def main() -> None:
    cfg = Config()
    set_global_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("=" * 70)
    print("FAS Memetic PPO Experiment (modular structure)")
    print(f"wavelength = {cfg.wavelength:.4e} m")
    print(
        f"W = {cfg.W_in_lambda:.4f} λ ({cfg.W:.4e} m), "
        f"Delta = {cfg.delta_port_in_lambda:.6f} λ ({cfg.delta_port:.4e} m)"
    )
    print(f"N={cfg.N}, M={cfg.M}, T={cfg.T}, SNR={cfg.snr_db} dB")
    print("=" * 70)

    model, callback = train_agent(cfg)
    eval_data = run_evaluation(cfg)

    print(f"Training episodes logged: {len(callback.episode_rewards)}")
    print(f"Evaluation average capacity: {np.mean(eval_data['capacities']):.4f} bps/Hz")
    print(f"Evaluation average active ports: {np.mean(eval_data['active_counts']):.4f}")
    print(f"Artifacts saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
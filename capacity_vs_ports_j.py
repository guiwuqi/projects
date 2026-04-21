import argparse
import os
from dataclasses import asdict
from typing import Dict, Iterable, List

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
    _, info = env.reset(seed=cfg.seed)
    clusters = env.clusters
    ula_caps = get_ula_capacity(cfg, cfg.ula_fixed_port_count, clusters)
    if len(ula_caps) == 0:
        raise RuntimeError("ULA capacity sequence is empty.")
    return float(np.mean(ula_caps))


def compute_curve(
    port_counts: Iterable[int],
    w_in_lambda: float,
    mode: str,
    model_path: str,
    base_cfg: Config,
    memetic_eval_only: bool = False,   # 新增参数
) -> List[float]:
    mode = mode.lower().strip()
    if mode not in {"ula", "fas_memetic", "fas_no_memetic"}:
        raise ValueError(f"Unsupported mode: {mode}")

    model = None
    if mode != "ula":
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(
                f"Model not found: {model_path}.zip\n"
                "Please train first or pass --model-path pointing to your PPO checkpoint (without .zip)."
            )
        model = PPO.load(model_path)

    ys: List[float] = []
    for k in port_counts:
        cfg = _build_cfg(**asdict(base_cfg))
        cfg.W_in_lambda = float(w_in_lambda)
        cfg.K = int(k)
        cfg.ula_fixed_port_count = int(k)

        if mode == "ula":
            y = evaluate_ula_avg_capacity(cfg)
        else:
            cfg.use_memetic_search = (mode == "fas_memetic")
            cfg.memetic_eval_only = memetic_eval_only   # 按训练时配置设定
            y = evaluate_fas_avg_capacity(cfg, model)

        ys.append(y)
        print(f"mode={mode:14s} W={w_in_lambda:<4} K={k:<3} avg_capacity={y:.4f}")

    return ys


def plot_figure_one(
    port_min: int,
    port_max: int,
    smppo_model: str,      # 新参数：S-MPPO 模型路径
    no_smppo_model: str,    # 新参数：no S-MPPO 模型路径
    out_path: str,
) -> None:
    if port_min < 1 or port_max < port_min:
        raise ValueError("Invalid port range.")

    base_cfg = Config()
    set_global_seed(base_cfg.seed)

    ks = list(range(port_min, port_max + 1))

    curves: Dict[str, List[float]] = {
        "ULA, W=3": compute_curve(ks, 3, "ula", "", base_cfg),   # ULA 不需要模型路径
        "FAS (S-MPPO), W=3": compute_curve(
            ks, 3, "fas_memetic", smppo_model, base_cfg, memetic_eval_only=False
        ),
        "FAS (no S-MPPO), W=3": compute_curve(
            ks, 3, "fas_no_memetic", no_smppo_model, base_cfg, memetic_eval_only=True
        ),
        # 以下为示例注释，可取消注释以绘制 W=6 的情况
        # "ULA, W=6": compute_curve(ks, 6, "ula", "", base_cfg),
        # "FAS (S-MPPO), W=6": compute_curve(ks, 6, "fas_memetic", smppo_model, base_cfg, memetic_eval_only=False),
    }

    plt.figure(figsize=(9.5, 5.5))
    for label, y in curves.items():
        linestyle = '--' if "ULA" in label else '-'
        plt.plot(ks, y, marker="o", linewidth=1.8, markersize=4, label=label, linestyle=linestyle)

    plt.xlabel("Number of active ports ($Q_{\\text{sub}}$)")
    plt.ylabel("Average channel capacity (bps/Hz)")
    plt.xlim(2, 20)
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
    parser = argparse.ArgumentParser(description="Plot Figure 1: average capacity vs active-port count.")
    parser.add_argument("--port-min", type=int, default=2, help="Minimum K (inclusive).")
    parser.add_argument("--port-max", type=int, default=20, help="Maximum K (inclusive).")
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
        default="comparison_plots/figure1_capacity_vs_ports.svg",
        help="Output figure path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_figure_one(
        port_min=args.port_min,
        port_max=args.port_max,
        smppo_model=args.smppo_model,
        no_smppo_model=args.no_smppo_model,
        out_path=args.out,
    )
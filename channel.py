from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg as la
from scipy.stats import vonmises

from config import Config


def set_global_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_uav_trajectory(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    times = np.arange(cfg.T) * cfg.dt
    az_deg, el_deg = cfg.uav_direction_deg
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)

    direction = np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=float,
    )
    start = np.array(cfg.uav_start, dtype=float)
    positions = np.array([start + cfg.uav_speed * t * direction for t in times], dtype=float)
    return times, positions


def build_tx_array_offsets(cfg: Config) -> np.ndarray:
    d = cfg.wavelength / 2.0
    idx = np.arange(cfg.M) - (cfg.M - 1) / 2.0
    offsets = np.zeros((cfg.M, 3), dtype=float)
    offsets[:, 0] = idx * d
    return offsets


def build_rx_port_positions(cfg: Config) -> np.ndarray:
    xs = np.linspace(50.0, cfg.W, cfg.N)
    pos = np.zeros((cfg.N, 3), dtype=float)
    pos[:, 0] = xs
    return pos


def build_velocity_direction(cfg: Config) -> np.ndarray:
    az_deg, el_deg = cfg.uav_direction_deg
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    v_dir = np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=float,
    )
    return v_dir / (np.linalg.norm(v_dir) + 1e-12)


def doppler_phase(cfg: Config, directions: np.ndarray, time_s: float, k: float) -> np.ndarray:
    """
    Compute path-wise Doppler phase exp(j*2π/λ * v*t*cos(theta)).
    directions: (P,3) unit path directions.
    returns: (P,)
    """
    v_dir = build_velocity_direction(cfg)
    projection = directions @ v_dir  # cos(theta), shape (P,)
    return np.exp(1j * k * cfg.uav_speed * time_s * projection)


def build_path_directions(alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """
    Build 3D unit propagation directions from azimuth/elevation.
    """
    return np.stack(
        [
            np.cos(betas) * np.cos(alphas),
            np.cos(betas) * np.sin(alphas),
            np.sin(betas),
        ],
        axis=1,
    )


def sample_paths(cfg: Config, rng: np.random.Generator) -> List[Dict[str, np.ndarray]]:
    clusters: List[Dict[str, np.ndarray]] = []

    # 定义散射体分布的平均距离
    mean_scatterer_distance = 30.0  # 单位：米，根据场景调整

    for mu_az_deg, mu_el_deg in zip(cfg.cluster_mean_aoa_deg, cfg.cluster_mean_eoa_deg):
        mu = np.deg2rad(mu_az_deg)
        mu_el = np.deg2rad(mu_el_deg)

        # 生成路径角度
        alphas = vonmises.rvs(kappa=cfg.kappa, loc=mu, size=cfg.Npath, random_state=rng)
        betas = vonmises.rvs(kappa=cfg.kappa, loc=mu_el, size=cfg.Npath, random_state=rng)
        betas = np.clip(betas, -np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)

        # 生成散射体距离（对数正态分布，更符合实际）
        distances = mean_scatterer_distance * np.exp(0.5 * rng.normal(size=cfg.Npath))

        # 远场近似：路径长度采用时间不变项，不再由瞬时几何关系引起角度/距离耦合变化
        # 这里用 2*distance 近似 Tx->scatter + scatter->Rx 的总传播距离
        total_path_lengths = 2.0 * distances
        distance_phase = np.exp(-1j * 2.0 * np.pi / cfg.wavelength * total_path_lengths)

        # 生成路径增益
        gains = (rng.normal(size=cfg.Npath) + 1j * rng.normal(size=cfg.Npath)) / np.sqrt(2.0 * cfg.L * cfg.Npath)

        clusters.append({
            "alphas": alphas,
            "betas": betas,
            "gains": gains,
            "distance_phase": distance_phase,
        })

    return clusters


def compute_channel_matrix(
        cfg: Config,
        uav_positions: np.ndarray,
        rx_port_positions: np.ndarray,
        clusters: List[Dict[str, np.ndarray]],
) -> np.ndarray:
    """Precompute H_full[t] ∈ C^{N×M} with proper distance phase."""
    k = 2.0 * np.pi / cfg.wavelength
    tx_offsets = build_tx_array_offsets(cfg)
    rx_ref = np.mean(rx_port_positions, axis=0)  # RX中心
    rx_offsets = rx_port_positions - rx_ref[None, :]  # (N, 3)

    H_full = np.zeros((cfg.T, cfg.N, cfg.M), dtype=np.complex128)

    for t in range(cfg.T):
        H_t = np.zeros((cfg.N, cfg.M), dtype=np.complex128)
        doppler_t = float(t * cfg.dt)

        for cl in clusters:
            alphas = cl["alphas"]
            betas = cl["betas"]
            gains = cl["gains"]
            distance_phase = cl["distance_phase"]  # (P,) 时间不变路径相位
            # 方向向量（用于多普勒和几何相位）
            directions = build_path_directions(alphas, betas)  # (P, 3)

            # 接收端几何相位（相对RX中心的相位差）
            rx_phase = np.exp(-1j * k * (rx_offsets @ directions.T))  # (N, P)
            # 发射端几何相位（相对UAV中心的相位差）
            tx_phase = np.exp(-1j * k * (tx_offsets @ directions.T))  # (M, P)

            # 多普勒相位
            doppler = doppler_phase(cfg, directions, doppler_t, k)  # (P,)

            # 合并所有相位项
            weighted_tx = (gains * doppler * distance_phase)[:, None] * tx_phase.T
            H_t += rx_phase @ weighted_tx

        H_full[t] = H_t

    return H_full


def compute_capacity(H_active: np.ndarray, rho_linear: float, M: int) -> float:
    if H_active.shape[0] == 0:
        return 0.0

    gram = H_active @ H_active.conj().T
    eye = np.eye(H_active.shape[0], dtype=np.complex128)
    mat = eye + (rho_linear / M) * gram

    sign, logdet = np.linalg.slogdet(mat)
    if sign <= 0:
        eigvals = np.clip(la.eigvalsh(mat).real, 1e-12, None)
        return float(np.sum(np.log2(eigvals)))
    return float(logdet / np.log(2.0))
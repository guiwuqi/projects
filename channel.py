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
    xs = np.linspace(0.0, cfg.W, cfg.N)
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
    if len(cfg.cluster_mean_eoa_deg) != len(cfg.cluster_mean_aoa_deg):
        raise ValueError("cluster_mean_eoa_deg 与 cluster_mean_aoa_deg 长度必须一致。")

    for mu_az_deg, mu_el_deg in zip(cfg.cluster_mean_aoa_deg, cfg.cluster_mean_eoa_deg):
        mu = np.deg2rad(mu_az_deg)
        mu_el = np.deg2rad(mu_el_deg)
        alphas = vonmises.rvs(kappa=cfg.kappa, loc=mu, size=cfg.Npath, random_state=rng)
        betas = vonmises.rvs(kappa=cfg.kappa, loc=mu_el, size=cfg.Npath, random_state=rng)
        betas = np.clip(betas, -np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
        gains = (rng.normal(size=cfg.Npath) + 1j * rng.normal(size=cfg.Npath)) / np.sqrt(2.0 * cfg.L * cfg.Npath)
        clusters.append({"alphas": alphas, "betas": betas, "gains": gains})
    return clusters


def compute_channel_matrix(
    cfg: Config,
    uav_positions: np.ndarray,
    rx_port_positions: np.ndarray,
    clusters: List[Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Precompute H_full[t] ∈ C^{N×M}. RX port coordinates explicitly enter phase.
    """
    k = 2.0 * np.pi / cfg.wavelength
    tx_offsets = build_tx_array_offsets(cfg)

    H_full = np.zeros((cfg.T, cfg.N, cfg.M), dtype=np.complex128)

    for t in range(cfg.T):
        tx_positions = uav_positions[t][None, :] + tx_offsets  # (M,3)
        H_t = np.zeros((cfg.N, cfg.M), dtype=np.complex128)
        doppler_t = float(t * cfg.dt)

        for cl in clusters:
            alphas = cl["alphas"]
            betas = cl["betas"]
            gains = cl["gains"]

            directions = build_path_directions(alphas, betas)  # (P,3)
            rx_phase = np.exp(-1j * k * (rx_port_positions @ directions.T))  # (N, P)
            tx_phase = np.exp(-1j * k * (tx_positions @ directions.T))  # (M, P)
            doppler = doppler_phase(cfg, directions, doppler_t, k)  # (P,)

            # Path-wise weighting: (P, M) = (P, 1) * (P, M)
            weighted_tx = (gains * doppler)[:, None] * tx_phase.T
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
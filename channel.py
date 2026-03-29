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


def sample_paths(cfg: Config, rng: np.random.Generator) -> List[Dict[str, np.ndarray]]:
    clusters: List[Dict[str, np.ndarray]] = []
    for mu_deg in cfg.cluster_mean_aoa_deg:
        mu = np.deg2rad(mu_deg)
        alphas = vonmises.rvs(kappa=cfg.kappa, loc=mu, size=cfg.Npath, random_state=rng)
        gains = (rng.normal(size=cfg.Npath) + 1j * rng.normal(size=cfg.Npath)) / np.sqrt(2.0 * cfg.L * cfg.Npath)
        clusters.append({"alphas": alphas, "gains": gains})
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

        for cl in clusters:
            alphas = cl["alphas"]
            gains = cl["gains"]

            directions = np.stack([np.cos(alphas), np.sin(alphas), np.zeros_like(alphas)], axis=1)  # (P,3)
            rx_phase = np.exp(-1j * k * (rx_port_positions @ directions.T))  # (N, P)
            tx_phase = np.exp(-1j * k * (tx_positions @ directions.T))       # (M, P)

            # Path-wise weighting: (P, M) = (P, 1) * (P, M)
            weighted_tx = gains[:, None] * tx_phase.T
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
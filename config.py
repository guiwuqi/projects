from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:
    # Physical constants
    c: float = 3e8
    fc: float = 5e9

    # UAV transmitter
    M: int = 4
    Tend: float = 3.0
    T: int = 30
    uav_speed: float = 5.0
    uav_start: Tuple[float, float, float] = (0.0, 0.0, 10.0)
    uav_direction_deg: Tuple[float, float] = (45.0, 15.0)  # azimuth, elevation

    # FAS receiver
    N: int = 500
    W_in_lambda: float = 10.0
    Kmin: int = 2
    Kmax: int = 10
    # 初始激活端口策略:
    # - "random": 原始随机窗口采样
    # - "spread": 尽量分散
    # - "clustered": 尽量集中
    # - "manual": 使用 init_active_ports 显式指定
    init_port_strategy: str = "spread"
    # init_active_ports: Optional[Tuple[int, ...]] = None
    init_active_ports = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499)
    # clustered 模式的窗口半径（索引单位）
    init_cluster_radius: int = 50

    # NLOS channel
    L: int = 3
    Npath: int = 10
    kappa: float = 3.0
    cluster_mean_aoa_deg: Tuple[float, float, float] = (-30.0, 0.0, 30.0)

    # SNR
    snr_db: float = 10.0

    # RL
    seed: int = 42
    total_timesteps: int = 100_000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10

    # Memetic local search
    use_memetic_search: bool = True
    action_neighbor_radius: int = 5

    # Spatial constraint: <= 5Δ
    neighbor_radius_multiples_delta: float = 5.0

    # Reward: default pure capacity
    #lambda_port_penalty: float = 0.5

    # IO
    out_dir: str = "results_fas_memetic_ppo"

    @property
    def wavelength(self) -> float:
        return self.c / self.fc

    @property
    def dt(self) -> float:
        return self.Tend / self.T

    @property
    def W(self) -> float:
        return self.W_in_lambda * self.wavelength

    @property
    def delta_port_in_lambda(self) -> float:
        return self.W_in_lambda / (self.N - 1)

    @property
    def delta_port(self) -> float:
        return self.W / (self.N - 1)

    @property
    def rho_linear(self) -> float:
        return 10 ** (self.snr_db / 10.0)